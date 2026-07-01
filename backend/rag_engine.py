"""
RAGEngine — the reusable, long-lived RAG instance behind the API
================================================================
This is the object the API endpoints share. It is built ONCE at process startup
(see api.py's lifespan) and reused for every request, so the expensive pieces —
Pinecone client, the Gemini embedding client, the Docling converter, the Cohere
reranker, the chat LLM, and the assembled retrieval chain — are initialized a
single time rather than per request.

Why this is cleanly reusable
----------------------------
Both retrievers (dense + sparse) are server-side Pinecone queries, so the engine
holds NO mutable corpus in memory. There is nothing to rebuild on upload and
nothing to rehydrate on restart, and no lock is needed: an upload just upserts
to Pinecone, and the next query sees it live. The retrieval chain is therefore
built once in __init__ and never rebuilt.

Two operations map to the two endpoints:
    • ingest(path)   ── upload endpoint  (PDF → chunks → dense + sparse upsert)
    • answer(query)  ── query endpoint   (retrieve → rerank → structured answer)

The generation layer lives here too: retrieved+reranked chunks are formatted into
a grounded prompt and sent to Gemini in a single (non-streaming) call. Gemini
returns a structured object with two components — the direct `answer` to the
question and document-grounded `insights` / next steps — plus citations.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# ─────────────────────────────────────────────────────────────────────────────
# Path anchoring — must come before any sibling-module imports
# ─────────────────────────────────────────────────────────────────────────────
_BACKEND_DIR = Path(__file__).resolve().parent   # .../nergy-clean/backend
_PROJECT_ROOT = _BACKEND_DIR.parent              # .../nergy-clean

for _p in (str(_BACKEND_DIR), str(_PROJECT_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Load .env from the backend/ directory regardless of CWD.
load_dotenv(_BACKEND_DIR / ".env")

from langchain_google_genai import (  # noqa: E402
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI,
)
from langchain_core.documents import Document  # noqa: E402
from langchain_core.prompts import ChatPromptTemplate  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

import hybrid_search as hs  # noqa: E402


# Generation model — kept on Google to match the embedding stack (one provider,
# one key). gemini-2.5-flash is fast and plenty for grounded RAG.
GENERATION_MODEL = os.getenv("GENERATION_MODEL", "gemini-2.5-flash")


class RAGAnswer(BaseModel):
    """Structured generation output: all components returned in one shot."""

    answer: str = Field(
        description=(
            "The direct answer to the user's question, grounded ONLY in the "
            "provided context. Cite sources inline as [filename, p.X]. If the "
            "context does not contain the answer, say you don't know."
        )
    )
    insights: list[str] = Field(
        default_factory=list,
        description=(
            "Additional insights, implications, or concrete next steps derived "
            "from the documents. Each item is one short, self-contained point, "
            "grounded in the context and cited as [filename, p.X]. Empty if the "
            "context supports none."
        ),
    )
    next_questions: list[str] = Field(
        default_factory=list,
        description=(
            "2-4 suggested follow-up questions the user could ask next, each "
            "answerable from these same documents. Empty if none are natural."
        ),
    )


SYSTEM_PROMPT = (
    "You are a precise assistant that answers ONLY from the provided context.\n"
    "Rules:\n"
    "- Use only the information in the context below. Do not use outside knowledge.\n"
    "- If the context does not contain the answer, say you don't know.\n"
    "- Cite the source after each claim as [filename, p.X] using the citation\n"
    "  tags shown with each context block.\n"
    "- Be concise and factual.\n"
    "Return a structured object with three components:\n"
    "  1. `answer`         — the direct answer to the question.\n"
    "  2. `insights`       — a list of document-grounded insights or next steps\n"
    "     that go beyond the literal answer (implications, related findings,\n"
    "     actions the documents suggest). Leave empty if the context supports none.\n"
    "  3. `next_questions` — a few natural follow-up questions the user could ask\n"
    "     next, each answerable from these same documents."
)

PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        ("human", "Context:\n{context}\n\nQuestion: {question}"),
    ]
)


def _page_label(meta: dict) -> str:
    ps, pe = meta.get("page_start", 0), meta.get("page_end", 0)
    return f"p.{ps}" if ps == pe else f"pp.{ps}-{pe}"


def _format_context(docs: list[Document]) -> str:
    """Render retrieved chunks into a numbered, citation-tagged context block."""
    blocks = []
    for i, doc in enumerate(docs, 1):
        meta = doc.metadata
        tag = f"[{meta.get('filename', '?')}, {_page_label(meta)}]"
        blocks.append(f"[{i}] {tag}\n{doc.page_content}")
    return "\n\n".join(blocks)


def _page_number(meta: dict) -> str:
    """Human-friendly page label for a citation: '5' or '5-6'."""
    ps, pe = meta.get("page_start", 0), meta.get("page_end", 0)
    return f"{ps}" if ps == pe else f"{ps}-{pe}"


def _citations(docs: list[Document]) -> list[dict]:
    """Structured citation payload sent to the client after the answer."""
    out = []
    for i, doc in enumerate(docs, 1):
        meta = doc.metadata
        out.append(
            {
                "n": i,
                "filename": meta.get("filename", ""),
                "section": meta.get("section", ""),
                "page": _page_number(meta),
                "page_start": meta.get("page_start", 0),
                "page_end": meta.get("page_end", 0),
                "pages": meta.get("pages", []),
                "rerank_score": meta.get("rerank_score"),
            }
        )
    return out


class RAGEngine:
    """Long-lived RAG instance shared across all API requests."""

    def __init__(self) -> None:
        self._validate_env()

        # ── Heavy, long-lived clients (built ONCE) ───────────────────────
        self.pc = hs.ensure_pinecone_index()
        self.embeddings = GoogleGenerativeAIEmbeddings(model=hs.EMBEDDING_MODEL)
        self.llm = ChatGoogleGenerativeAI(
            model=GENERATION_MODEL,
            temperature=0.1,
        )
        # Shared Docling converter so each upload doesn't re-init the models.
        self.converter = hs._make_docling_converter()

        # ── The stateless retrieval chain (built ONCE, reused forever) ───
        self.retrieval_chain = hs.build_retrieval_chain(self.pc, self.embeddings)

        # ── The generation chain (prompt → LLM → structured RAGAnswer) ───
        self.generation_chain = PROMPT | self.llm.with_structured_output(RAGAnswer)

        hs.UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _validate_env() -> None:
        missing = [
            k
            for k in ("PINECONE_API_KEY", "GOOGLE_API_KEY", "COHERE_API_KEY")
            if not os.getenv(k)
        ]
        if missing:
            raise RuntimeError(f"Missing required env vars: {', '.join(missing)}")

    # ── Upload endpoint ──────────────────────────────────────────────────
    def ingest(self, pdf_path: str | Path) -> dict:
        """
        Ingest one PDF: parse → chunk → upsert to dense + sparse indexes.

        Synchronous + CPU/IO-heavy (Docling parse, embeddings). The API layer
        runs it in a threadpool so it doesn't block the event loop. Idempotent:
        deterministic chunk IDs mean re-uploading the same file overwrites.
        """
        pdf_path = Path(pdf_path)
        name, dl_doc = hs.convert_pdf(pdf_path, converter=self.converter)
        chunks = hs.chunk_documents([(name, dl_doc)])
        if not chunks:
            return {"filename": name, "chunks_added": 0}

        hs.index_corpus(self.pc, chunks, self.embeddings)
        return {"filename": name, "chunks_added": len(chunks)}

    # ── Query endpoint ─────────────────────────────────────────────────────
    def retrieve(self, query: str) -> list[Document]:
        """Run the (sync) retrieval+rerank chain. Returns the final top-K docs."""
        return self.retrieval_chain.invoke(query)

    async def answer(self, query: str) -> dict:
        """
        Answer a query in one shot (no streaming) and return the whole result:

            {
                "answer":         "<direct grounded answer>",
                "insights":       ["<insight / next step>", ...],
                "next_questions": ["<suggested follow-up>", ...],
                "citations":      [...],
            }

        Retrieval and generation are both synchronous/network-bound, so we offload
        them to a thread to avoid blocking the event loop. Generation returns a
        structured RAGAnswer with the two components rather than a token stream.
        """
        # 1) Retrieve + rerank (sync work off the event loop).
        docs = await asyncio.to_thread(self.retrieve, query)

        if not docs:
            return {
                "answer": "I don't have any indexed documents to answer from.",
                "insights": [],
                "next_questions": [],
                "citations": [],
            }

        # 2) Generate the grounded, structured answer in a single call.
        context = _format_context(docs)
        result: RAGAnswer = await self.generation_chain.ainvoke(
            {"context": context, "question": query}
        )

        return {
            "answer": result.answer,
            "insights": result.insights,
            "next_questions": result.next_questions,
            "citations": _citations(docs),
        }
