"""
Hybrid Search RAG — LangChain Parallel Chain
=============================================
Dense (Pinecone SDK + Google Embeddings) + Sparse (BM25) retrieval
using LangChain's RunnableParallel for parallel execution and
RunnableLambda for RRF fusion, followed by a Cohere cross-encoder
rerank step that narrows the fused candidates to the final results.

Ingest: Docling (PDF→DoclingDocument) → HybridChunker (structure + token aware,
        page-tracked) → Gemini embeddings → Pinecone.
Query:  retrieve wide (RETRIEVE_K each) → RRF fuse → Cohere rerank → top FINAL_K

PDFs are loaded from the ./uploads folder.
Run:  python hybrid_search.py
"""

from __future__ import annotations

import hashlib
import os
import sys
import time
from pathlib import Path

import cohere
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    AcceleratorOptions,
    AcceleratorDevice,
)
from docling.datamodel.document import DoclingDocument
from docling.chunking import HybridChunker
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from transformers import AutoTokenizer

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document
from langchain_core.runnables import (
    RunnableParallel,
    RunnableLambda,
    RunnablePassthrough,
)

# ─────────────────────────────────────────────────────────────────────────────
# Path anchoring
# ─────────────────────────────────────────────────────────────────────────────
# Resolve the backend/ directory relative to THIS file so all paths are correct
# regardless of the CWD from which uvicorn / python is launched.
BACKEND_DIR = Path(__file__).resolve().parent    # .../nergy-clean/backend
_PROJECT_ROOT = BACKEND_DIR.parent               # .../nergy-clean

for _p in (str(BACKEND_DIR), str(_PROJECT_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Load .env from the backend/ directory regardless of CWD.
load_dotenv(BACKEND_DIR / ".env")

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

# Uploads folder lives inside backend/ — resolved from __file__ so it is
# always correct even when uvicorn is started from the project root.
UPLOADS_DIR = BACKEND_DIR / "uploads"
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
# Dense index — our own Gemini embeddings (cosine, 3072-dim).
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "rag-hybrid-search")
# Sparse index — server-side keyword/lexical half, replacing the old in-memory
# BM25 retriever. This is a BYO-sparse-vectors index (vector_type='sparse',
# metric='dotproduct', NO integrated model): we generate the sparse vectors
# ourselves with Pinecone's hosted sparse model (SPARSE_MODEL via the inference
# API) and upsert/query them with `sparse_values`. Moving sparse server-side is
# what makes the engine stateless: nothing to rebuild in memory on upload.
PINECONE_SPARSE_INDEX_NAME = os.getenv("PINECONE_SPARSE_INDEX_NAME", "sprase-index")
COHERE_API_KEY = os.getenv("COHERE_API_KEY", "")
EMBEDDING_MODEL = "gemini-embedding-2-preview"
# gemini-embedding-2-preview returns 3072-dim vectors by default (native,
# already L2-normalized). The Pinecone index dimension MUST match this.
EMBEDDING_DIM = 3072
# Pinecone-hosted sparse embedding model (DeepImpact-style learned sparse).
# 512-token max sequence — lines up with our MAX_TOKENS=512 chunking.
SPARSE_MODEL = "pinecone-sparse-english-v0"
# Pinecone inference embed accepts at most 96 inputs per call.
SPARSE_EMBED_BATCH = 96
RERANK_MODEL = "rerank-v3.5"
# Chunking: Docling HybridChunker runs on the DoclingDocument directly (NOT
# exported Markdown), so each chunk keeps its page provenance for citations.
# It is tokenization-aware: it splits chunks over MAX_TOKENS and merges
# undersized adjacent chunks that share the same headings (merge_peers) — the
# automatic version of the old hand-tuned "orphan tail" merge.
#
# CAVEAT: Gemini's tokenizer isn't public, so we size with an HF tokenizer as a
# proxy — token counts are APPROXIMATE. MAX_TOKENS (512) sits well under the
# gemini-embedding input limit (~2048 tokens), leaving margin for the heading
# context that contextualize() prepends, and keeps feature lists / equations in
# one chunk (≈ the old 1920-char whole-section target).
TOKENIZER_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MAX_TOKENS = 512

# Retrieval funnel: pull a WIDE candidate set from each retriever, fuse with RRF,
# then let the cross-encoder reranker narrow it down to the FINAL set.
# Reranking only adds value when RETRIEVE_K >> FINAL_K.
RETRIEVE_K = 30   # candidates per retriever + size of the fused list handed to the reranker
FINAL_K = 5       # docs kept after reranking

DENSE_WEIGHT = 0.7
SPARSE_WEIGHT = 0.3


# ─────────────────────────────────────────────────────────────────────────────
# PDF Loading — Docling: layout-aware parsing → clean Markdown
# ─────────────────────────────────────────────────────────────────────────────
def _make_docling_converter() -> DocumentConverter:
    """
    Docling converter tuned for TEXT PDFs:
      • OCR disabled — we only handle digital/text PDFs, so skip the image
        models (much faster, smaller footprint).
      • Table-structure recognition ON — academic tables export as proper
        Markdown tables instead of scrambled text.
    """
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = False
    pipeline_options.do_table_structure = True
    # Force CPU — this box's CUDA/cuDNN isn't initialized, and we don't need a
    # GPU for text-only digital PDFs.
    pipeline_options.accelerator_options = AcceleratorOptions(
        device=AcceleratorDevice.CPU
    )

    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )


def convert_pdf(
    pdf_path: Path,
    converter: DocumentConverter | None = None,
) -> tuple[str, DoclingDocument]:
    """
    Parse a SINGLE PDF into (filename, DoclingDocument).

    Used by the API's upload endpoint, which ingests one file at a time. Pass a
    shared `converter` to avoid re-initializing Docling's models per call.
    """
    converter = converter or _make_docling_converter()
    result = converter.convert(str(pdf_path))
    return pdf_path.name, result.document


def load_pdfs(uploads_dir: Path = UPLOADS_DIR) -> list[tuple[str, DoclingDocument]]:
    """
    Parse each PDF into a DoclingDocument via Docling.

    Returns a list of (filename, DoclingDocument) tuples. We deliberately keep
    the structured DoclingDocument (NOT exported Markdown): the chunker runs on
    it directly so every chunk retains page provenance (page_no / bbox) for
    citations. Docling also reconstructs the real reading order (fixing
    multi-column interleaving) and recovers the heading hierarchy.
    """
    pdf_files = list(uploads_dir.glob("*.pdf"))
    if not pdf_files:
        print(f"  [WARN] No PDF files found in {uploads_dir.resolve()}")
        return []

    converter = _make_docling_converter()
    dl_docs: list[tuple[str, DoclingDocument]] = []

    for pdf_path in pdf_files:
        print(f"  Converting (Docling): {pdf_path.name}")
        result = converter.convert(str(pdf_path))
        dl_docs.append((pdf_path.name, result.document))

    print(f"  Converted {len(dl_docs)} PDF(s).")
    return dl_docs


# ─────────────────────────────────────────────────────────────────────────────
# Chunking — Docling HybridChunker: structure-aware + token-aware + page-tracked
# ─────────────────────────────────────────────────────────────────────────────
def _make_hybrid_chunker() -> HybridChunker:
    """
    HybridChunker = HierarchicalChunker (group doc items by their section) plus
    two token-aware passes: split any chunk over MAX_TOKENS, then merge_peers
    (combine undersized adjacent chunks that share the same headings).
    Sized with an HF tokenizer as a proxy for Gemini's (approximate; see config).
    """
    tokenizer = HuggingFaceTokenizer(
        tokenizer=AutoTokenizer.from_pretrained(TOKENIZER_MODEL),
        max_tokens=MAX_TOKENS,
    )
    return HybridChunker(tokenizer=tokenizer, merge_peers=True)


def _chunk_pages(chunk) -> list[int]:
    """Sorted, de-duplicated page numbers this chunk spans (from prov)."""
    return sorted(
        {prov.page_no for item in chunk.meta.doc_items for prov in item.prov}
    )


def chunk_documents(dl_docs: list[tuple[str, DoclingDocument]]) -> list[Document]:
    """
    Chunk each DoclingDocument with HybridChunker.

    For every chunk we keep:
      • page_content = chunker.contextualize(chunk) — the chunk text with its
        heading/caption context prepended, so the embedding carries section
        identity (replaces the old hand-built breadcrumb prefix).
      • section    — 'file > Heading > Subheading' breadcrumb, for display.
      • page_start / page_end / pages — page provenance for citations. A chunk
        can span multiple pages, so we record the full range.
    """
    chunker = _make_hybrid_chunker()

    all_chunks: list[Document] = []
    for filename, dl_doc in dl_docs:
        for chunk in chunker.chunk(dl_doc=dl_doc):
            headings = chunk.meta.headings or []
            breadcrumb = " > ".join([filename, *headings])
            pages = _chunk_pages(chunk)

            metadata = {
                "filename": filename,
                "section": breadcrumb,
                # Pinecone metadata must be flat scalars (or list[str]).
                "page_start": pages[0] if pages else 0,
                "page_end": pages[-1] if pages else 0,
                "pages": [str(p) for p in pages],
            }
            all_chunks.append(
                Document(
                    page_content=chunker.contextualize(chunk=chunk),
                    metadata=metadata,
                )
            )

    print(f"  Total chunks: {len(all_chunks)}")
    return all_chunks


# ─────────────────────────────────────────────────────────────────────────────
# Pinecone index setup + upsert
# ─────────────────────────────────────────────────────────────────────────────
def ensure_pinecone_index() -> Pinecone:
    """
    Ensure both indexes are reachable:
      • dense  ('cosine', 3072-dim) — created here if missing.
      • sparse ('dotproduct', BYO sparse vectors) — must already exist; we do
        NOT auto-create it because it was provisioned with custom settings in
        the console. Raise a clear error if it's absent instead of silently
        creating one with the wrong config.
    """
    pc = Pinecone(api_key=PINECONE_API_KEY)
    existing = [idx.name for idx in pc.list_indexes()]

    # ── Dense index ──────────────────────────────────────────────────────
    if PINECONE_INDEX_NAME not in existing:
        print(f"  Creating dense Pinecone index '{PINECONE_INDEX_NAME}' …")
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=EMBEDDING_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        while not pc.describe_index(PINECONE_INDEX_NAME).status.get("ready"):
            time.sleep(1)
        print(f"  Dense index ready.")
    else:
        print(f"  Dense index '{PINECONE_INDEX_NAME}' exists.")

    # ── Sparse index ─────────────────────────────────────────────────────
    if PINECONE_SPARSE_INDEX_NAME not in existing:
        raise RuntimeError(
            f"Sparse index '{PINECONE_SPARSE_INDEX_NAME}' not found. Create it in "
            f"Pinecone as vector_type='sparse', metric='dotproduct' (no embedding "
            f"model), then set PINECONE_SPARSE_INDEX_NAME."
        )
    print(f"  Sparse index '{PINECONE_SPARSE_INDEX_NAME}' exists.")

    return pc


def make_chunk_id(chunk: Document) -> str:
    """
    Deterministic vector ID derived from the chunk's source + content.

    The same chunk always maps to the same ID, so re-running indexing
    *overwrites* existing vectors instead of inserting duplicates with fresh
    random IDs. ID = sha1("filename|page|content").
    """
    filename = chunk.metadata.get("filename", "")
    section = chunk.metadata.get("section", "")
    raw = f"{filename}|{section}|{chunk.page_content}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def index_chunks_to_pinecone(
    pc: Pinecone,
    chunks: list[Document],
    embeddings: GoogleGenerativeAIEmbeddings,
    batch_size: int = 100,
) -> None:
    """Embed chunks and upsert into Pinecone in batches.

    Uses deterministic per-chunk IDs (see make_chunk_id), so indexing is
    idempotent: re-running on the same documents overwrites the existing
    vectors rather than accumulating duplicates.
    """
    index = pc.Index(PINECONE_INDEX_NAME)
    texts = [chunk.page_content for chunk in chunks]

    print(f"  Embedding {len(texts)} chunks …")
    all_vectors = embeddings.embed_documents(texts)

    # Upsert in batches
    for i in range(0, len(chunks), batch_size):
        batch_chunks = chunks[i : i + batch_size]
        batch_vectors = all_vectors[i : i + batch_size]

        vectors_to_upsert = []
        for chunk, vector in zip(batch_chunks, batch_vectors):
            vec_id = make_chunk_id(chunk)
            # Pinecone metadata must be flat (scalars or list[str])
            metadata = {
                "text": chunk.page_content,
                "filename": chunk.metadata.get("filename", ""),
                "section": chunk.metadata.get("section", ""),
                "page_start": chunk.metadata.get("page_start", 0),
                "page_end": chunk.metadata.get("page_end", 0),
                "pages": chunk.metadata.get("pages", []),
            }
            vectors_to_upsert.append((vec_id, vector, metadata))

        index.upsert(vectors=vectors_to_upsert)

    print(f"  Upserted {len(chunks)} dense vectors to '{PINECONE_INDEX_NAME}'.")


# ─────────────────────────────────────────────────────────────────────────────
# Sparse vectors — Pinecone hosted sparse model via the inference API
# ─────────────────────────────────────────────────────────────────────────────
def embed_sparse(pc: Pinecone, texts: list[str], input_type: str) -> list[dict]:
    """
    Embed `texts` with Pinecone's hosted sparse model and return a list of
    {"indices": [...], "values": [...]} dicts ready for upsert / query.

    input_type is 'passage' when indexing documents and 'query' when searching
    — the model encodes the two asymmetrically. Batched at SPARSE_EMBED_BATCH
    (the inference API caps inputs per call).
    """
    out: list[dict] = []
    for i in range(0, len(texts), SPARSE_EMBED_BATCH):
        batch = texts[i : i + SPARSE_EMBED_BATCH]
        resp = pc.inference.embed(
            model=SPARSE_MODEL,
            inputs=batch,
            parameters={"input_type": input_type, "truncate": "END"},
        )
        for emb in resp.data:
            out.append(
                {"indices": emb.sparse_indices, "values": emb.sparse_values}
            )
    return out


def index_sparse_to_pinecone(
    pc: Pinecone,
    chunks: list[Document],
    batch_size: int = 100,
) -> None:
    """Embed chunks as sparse vectors and upsert into the sparse index.

    Mirrors index_chunks_to_pinecone (same deterministic IDs, same metadata) but
    writes `sparse_values` instead of a dense vector. Idempotent for the same
    reason: re-running overwrites by ID rather than duplicating.
    """
    index = pc.Index(PINECONE_SPARSE_INDEX_NAME)
    texts = [chunk.page_content for chunk in chunks]

    print(f"  Sparse-embedding {len(texts)} chunks …")
    sparse_vecs = embed_sparse(pc, texts, input_type="passage")

    for i in range(0, len(chunks), batch_size):
        batch_chunks = chunks[i : i + batch_size]
        batch_sparse = sparse_vecs[i : i + batch_size]

        vectors_to_upsert = []
        for chunk, sparse in zip(batch_chunks, batch_sparse):
            vec_id = make_chunk_id(chunk)
            metadata = {
                "text": chunk.page_content,
                "filename": chunk.metadata.get("filename", ""),
                "section": chunk.metadata.get("section", ""),
                "page_start": chunk.metadata.get("page_start", 0),
                "page_end": chunk.metadata.get("page_end", 0),
                "pages": chunk.metadata.get("pages", []),
            }
            vectors_to_upsert.append(
                {"id": vec_id, "sparse_values": sparse, "metadata": metadata}
            )

        index.upsert(vectors=vectors_to_upsert)

    print(f"  Upserted {len(chunks)} sparse vectors to '{PINECONE_SPARSE_INDEX_NAME}'.")


# ─────────────────────────────────────────────────────────────────────────────
# Dense retriever — Pinecone SDK wrapped as a callable for LCEL
# ─────────────────────────────────────────────────────────────────────────────
def make_dense_retriever_fn(
    pc: Pinecone,
    embeddings: GoogleGenerativeAIEmbeddings,
    top_k: int = RETRIEVE_K,
):
    """
    Returns a function: query (str) -> list[Document]
    that queries Pinecone directly using the SDK.
    """
    index = pc.Index(PINECONE_INDEX_NAME)

    def retrieve(query: str) -> list[Document]:
        query_vector = embeddings.embed_query(query)
        results = index.query(
            vector=query_vector,
            top_k=top_k,
            include_metadata=True,
        )
        docs = []
        for match in results.matches:
            doc = Document(
                page_content=match.metadata.get("text", ""),
                metadata={
                    "filename": match.metadata.get("filename", ""),
                    "section": match.metadata.get("section", ""),
                    "page_start": match.metadata.get("page_start", 0),
                    "page_end": match.metadata.get("page_end", 0),
                    "pages": match.metadata.get("pages", []),
                    "dense_score": round(match.score, 6),
                },
            )
            docs.append(doc)
        return docs

    return retrieve


# ─────────────────────────────────────────────────────────────────────────────
# Sparse retriever — Pinecone sparse index, the lexical half (replaces BM25)
# ─────────────────────────────────────────────────────────────────────────────
def make_sparse_retriever_fn(
    pc: Pinecone,
    top_k: int = RETRIEVE_K,
):
    """
    Returns a function: query (str) -> list[Document]

    The server-side counterpart of the old in-memory BM25Retriever: it sparse-
    embeds the query (input_type='query') and queries the sparse Pinecone index
    with `sparse_vector`. Returns Documents shaped exactly like the dense
    retriever's so RRF fusion can merge the two on page_content.
    """
    index = pc.Index(PINECONE_SPARSE_INDEX_NAME)

    def retrieve(query: str) -> list[Document]:
        sparse = embed_sparse(pc, [query], input_type="query")[0]
        results = index.query(
            sparse_vector=sparse,
            top_k=top_k,
            include_metadata=True,
        )
        docs = []
        for match in results.matches:
            doc = Document(
                page_content=match.metadata.get("text", ""),
                metadata={
                    "filename": match.metadata.get("filename", ""),
                    "section": match.metadata.get("section", ""),
                    "page_start": match.metadata.get("page_start", 0),
                    "page_end": match.metadata.get("page_end", 0),
                    "pages": match.metadata.get("pages", []),
                    "sparse_score": round(match.score, 6),
                },
            )
            docs.append(doc)
        return docs

    return retrieve


# ─────────────────────────────────────────────────────────────────────────────
# RRF Fusion — the merge step in the chain
# ─────────────────────────────────────────────────────────────────────────────
def rrf_fusion(
    parallel_results: dict[str, list[Document]],
    weights: dict[str, float] | None = None,
    k: int = 60,
    top_n: int = RETRIEVE_K,
) -> list[Document]:
    """
    Reciprocal Rank Fusion.
    Receives {"dense": [...], "sparse": [...]} from RunnableParallel,
    fuses them and returns top_n ranked docs. top_n defaults to the WIDE
    candidate count (RETRIEVE_K) so the reranker has a rich set to work with.
    """
    if weights is None:
        weights = {"dense": DENSE_WEIGHT, "sparse": SPARSE_WEIGHT}

    fused_scores: dict[str, float] = {}
    doc_map: dict[str, Document] = {}

    for key, docs in parallel_results.items():
        w = weights.get(key, 0.5)
        for rank, doc in enumerate(docs):
            doc_key = doc.page_content
            rrf_score = w * (1.0 / (k + rank + 1))
            fused_scores[doc_key] = fused_scores.get(doc_key, 0.0) + rrf_score
            doc_map[doc_key] = doc

    sorted_keys = sorted(fused_scores, key=lambda c: fused_scores[c], reverse=True)

    ranked = []
    for content_key in sorted_keys:
        doc = doc_map[content_key]
        doc.metadata["rrf_score"] = round(fused_scores[content_key], 6)
        ranked.append(doc)

    return ranked[:top_n]


# ─────────────────────────────────────────────────────────────────────────────
# Reranking — Cohere cross-encoder, the precision step after fusion
# ─────────────────────────────────────────────────────────────────────────────
def make_reranker_fn(top_n: int = FINAL_K):
    """
    Returns a function: {"query": str, "docs": [Document]} -> list[Document]

    Sends the fused candidate list to Cohere's cross-encoder, which scores each
    (query, document) pair jointly (unlike the bi-encoder embeddings used for
    dense retrieval) and returns the top_n most relevant docs, re-sorted, each
    tagged with a `rerank_score`. If Cohere errors, we fall back to the fused
    order so a transient API issue degrades gracefully instead of crashing.
    """
    client = cohere.ClientV2(api_key=COHERE_API_KEY)

    def rerank(payload: dict) -> list[Document]:
        query: str = payload["query"]
        docs: list[Document] = payload["docs"]

        if not docs:
            return []

        try:
            response = client.rerank(
                model=RERANK_MODEL,
                query=query,
                documents=[d.page_content for d in docs],
                top_n=min(top_n, len(docs)),
            )
        except Exception as exc:  # noqa: BLE001 — degrade to fused order
            print(f"  [WARN] Cohere rerank failed ({exc}); using fused order.")
            return docs[:top_n]

        reranked: list[Document] = []
        for result in response.results:
            doc = docs[result.index]
            doc.metadata["rerank_score"] = round(result.relevance_score, 6)
            reranked.append(doc)
        return reranked

    return rerank


# ─────────────────────────────────────────────────────────────────────────────
# Build the hybrid search chain (LCEL)
# ─────────────────────────────────────────────────────────────────────────────
def build_retrieval_chain(
    pc: Pinecone,
    embeddings: GoogleGenerativeAIEmbeddings,
    retrieve_k: int = RETRIEVE_K,
    final_k: int = FINAL_K,
):
    """
    Build the retrieval LCEL chain over ALREADY-INDEXED corpora.

        query (str)
          │
          ├──► dense  (Pinecone dense index)  ──┐
          │                                      ├──► rrf_fusion ──┐
          └──► sparse (Pinecone sparse index) ──┘  (top RETRIEVE_K) │
          │                                                          ├──► Cohere rerank ──► top FINAL_K
          └──────────────── query (passthrough) ─────────────────────┘

    BOTH retrievers are now server-side Pinecone queries, so this chain is
    completely stateless — no in-memory corpus. That is what lets the API engine
    build it ONCE and reuse it across every request; newly uploaded documents
    show up automatically because the query hits Pinecone live (no rebuild). The
    original query is carried alongside the docs so the reranker can score
    (query, doc) pairs.
    """
    # Dense retriever — Pinecone SDK wrapped as RunnableLambda
    dense_retriever = RunnableLambda(make_dense_retriever_fn(pc, embeddings, retrieve_k))

    # Sparse retriever — Pinecone sparse index (replaces in-memory BM25)
    sparse_retriever = RunnableLambda(make_sparse_retriever_fn(pc, retrieve_k))

    # Retrieve wide + fuse → a single fused candidate list
    fusion_chain = (
        RunnableParallel(
            dense=dense_retriever,
            sparse=sparse_retriever,
        )
        | RunnableLambda(rrf_fusion)
    )

    # Cohere reranker — narrows the fused list to the final top_n
    reranker = RunnableLambda(make_reranker_fn(final_k))

    # ── LCEL Chain: fuse, carry the query alongside, then rerank ──────────
    hybrid_chain = (
        RunnableParallel(
            docs=fusion_chain,
            query=RunnablePassthrough(),
        )
        | reranker
    )

    return hybrid_chain


def index_corpus(
    pc: Pinecone,
    chunks: list[Document],
    embeddings: GoogleGenerativeAIEmbeddings,
) -> None:
    """Index chunks into BOTH Pinecone indexes (dense + sparse)."""
    index_chunks_to_pinecone(pc, chunks, embeddings)
    index_sparse_to_pinecone(pc, chunks)


def build_hybrid_chain(
    pc: Pinecone,
    chunks: list[Document],
    embeddings: GoogleGenerativeAIEmbeddings,
    retrieve_k: int = RETRIEVE_K,
    final_k: int = FINAL_K,
):
    """
    Index `chunks` into both indexes, then build the retrieval chain.

    Thin wrapper kept for the standalone CLI (main): it pays the upsert cost and
    then delegates to build_retrieval_chain. The API engine separates these two
    steps so it builds the chain once and only pays indexing on upload.
    """
    index_corpus(pc, chunks, embeddings)
    return build_retrieval_chain(pc, embeddings, retrieve_k, final_k)


# ─────────────────────────────────────────────────────────────────────────────
# Display
# ─────────────────────────────────────────────────────────────────────────────
def print_results(query: str, results: list[Document]) -> None:
    print(f"\n{'='*80}")
    print(f"  QUERY: {query}")
    print(f"{'='*80}")
    for i, doc in enumerate(results, 1):
        meta = doc.metadata
        print(f"\n  [{i}]  Rerank Score: {meta.get('rerank_score', 'N/A')}"
              f"   (RRF: {meta.get('rrf_score', 'N/A')})")
        ps, pe = meta.get("page_start", 0), meta.get("page_end", 0)
        page_str = f"p. {ps}" if ps == pe else f"pp. {ps}–{pe}"
        print(f"       File: {meta.get('filename', '?')}  ({page_str})")
        print(f"       Section: {meta.get('section', '?')}")
        print(f"       ─────────────────────────────────────────")
        preview = doc.page_content[:300].replace("\n", " ")
        if len(doc.page_content) > 300:
            preview += " …"
        print(f"       {preview}")
    print(f"\n{'='*80}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    if not PINECONE_API_KEY:
        print("[ERROR] PINECONE_API_KEY not set in .env")
        return
    if not os.getenv("GOOGLE_API_KEY"):
        print("[ERROR] GOOGLE_API_KEY not set in .env")
        return
    if not COHERE_API_KEY:
        print("[ERROR] COHERE_API_KEY not set in .env (needed for reranking)")
        return

    print("\n[1/4] Parsing PDFs (Docling) …")
    dl_docs = load_pdfs()
    if not dl_docs:
        print("  No documents to process. Add PDFs to the uploads/ folder and retry.")
        return

    print("\n[2/4] HybridChunker (page-tracked) chunking …")
    chunks = chunk_documents(dl_docs)

    print("\n[3/4] Setting up Pinecone + building chain …")
    pc = ensure_pinecone_index()
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)

    t0 = time.time()
    hybrid_chain = build_hybrid_chain(pc, chunks, embeddings)
    print(f"  Chain ready in {time.time() - t0:.1f}s")

    # ── Interactive query loop ───────────────────────────────────────────
    print("\n[4/4] Ready! Enter queries below (type 'quit' to exit)\n")
    while True:
        try:
            query = input("Query ▸ ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not query or query.lower() in ("quit", "exit", "q"):
            print("Exiting.")
            break

        t0 = time.time()
        results = hybrid_chain.invoke(query)
        elapsed = time.time() - t0

        print_results(query, results)
        print(f"  ⏱  Retrieved in {elapsed:.2f}s\n")


if __name__ == "__main__":
    main()
