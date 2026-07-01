"""
FastAPI surface for the hybrid-search RAG
=========================================
Two endpoints, one shared engine:

    POST /upload   — multipart PDF → ingest into dense + sparse Pinecone indexes
    POST /query    — {"query": "..."} → JSON {answer, insights, citations}
                     (retrieval layer + generation layer, single response)
    GET  /health   — liveness

Reusable-instance pattern
-------------------------
The RAGEngine is built ONCE in the lifespan handler and stored on app.state.
Every request gets the same instance via the get_engine dependency — no per
request re-initialization of Pinecone / embeddings / Docling / the LLM.

Run:  uvicorn api:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import asyncio
import shutil
import sys
from contextlib import asynccontextmanager
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# Path anchoring — must come before sibling-module imports
# ─────────────────────────────────────────────────────────────────────────────
# Resolve backend/ from __file__ so uvicorn can be launched from any directory
# (e.g. from the project root: uvicorn backend.api:app).
_BACKEND_DIR = Path(__file__).resolve().parent
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

from fastapi import Depends, FastAPI, HTTPException, Request, UploadFile, File  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from pydantic import AliasChoices, BaseModel, Field  # noqa: E402

import hybrid_search as hs  # noqa: E402
from rag_engine import RAGEngine  # noqa: E402



@asynccontextmanager
async def lifespan(app: FastAPI):
    # Build the single, shared engine at startup (loads models, clients, chain).
    app.state.engine = RAGEngine()
    yield
    # Nothing to tear down — clients are stateless network handles.


app = FastAPI(title="Hybrid Search RAG", version="1.0.0", lifespan=lifespan)

# ── CORS ──────────────────────────────────────────────────────────────────────
# The frontend is a static index.html that may be opened directly (file:// →
# Origin "null") or served from any localhost port. Allow all origins for local
# dev. The app uses no cookies/credentials, so credentials stay off — the CORS
# spec forbids "*" together with allow_credentials=True. Tighten in production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_engine(request: Request) -> RAGEngine:
    """Dependency that hands every endpoint the one shared engine instance."""
    return request.app.state.engine


class QueryRequest(BaseModel):
    # The frontend posts {"question": "..."}; accept "query" as an alias too.
    question: str = Field(validation_alias=AliasChoices("question", "query"))


# ─────────────────────────────────────────────────────────────────────────────
# Upload endpoint
# ─────────────────────────────────────────────────────────────────────────────
@app.post("/upload")
async def upload(
    file: UploadFile = File(...),
    engine: RAGEngine = Depends(get_engine),
):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only .pdf files are supported.")

    # Persist to uploads/ (also the source of truth for the corpus on disk).
    dest = hs.UPLOADS_DIR / Path(file.filename).name
    with dest.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    # Ingestion is sync + heavy → run off the event loop.
    try:
        result = await asyncio.to_thread(engine.ingest, dest)
    except Exception as exc:  # surface a clean error to the client
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {exc}")

    return {"status": "ok", **result}


















# ─────────────────────────────────────────────────────────────────────────────
# Query endpoint — single JSON response (no streaming)
# ─────────────────────────────────────────────────────────────────────────────
@app.post("/query")
async def query(
    body: QueryRequest,
    engine: RAGEngine = Depends(get_engine),
):
    q = body.question.strip()
    if not q:
        raise HTTPException(status_code=400, detail="question must not be empty.")

    try:
        result = await engine.answer(q)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Query failed: {exc}")

    # {"answer": "...", "insights": [...], "citations": [...]}
    return result


@app.get("/health")
async def health():
    return {"status": "ok"}
