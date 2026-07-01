# API Endpoint Flowcharts

Mermaid flowcharts for the two application endpoints. `GET /health` (liveness) is
omitted intentionally.

## `POST /upload` — ingest a PDF

```mermaid
flowchart TD
    A["Client<br/>POST /upload (multipart PDF)"] --> B{"filename ends<br/>with .pdf?"}
    B -- no --> B1["HTTP 400<br/>Only .pdf files supported"]
    B -- yes --> C["Save file to uploads/"]
    C --> D["engine.ingest(path)<br/>(run off event loop via threadpool)"]

    subgraph ingest ["RAGEngine.ingest"]
        D --> E["convert_pdf<br/>Docling parse → document"]
        E --> F["chunk_documents<br/>HybridChunker → page-tracked chunks"]
        F --> G{"any chunks?"}
        G -- no --> G1["chunks_added = 0"]
        G -- yes --> H["index_corpus<br/>embed + upsert to Pinecone<br/>(dense + sparse)"]
    end

    H --> I["HTTP 200<br/>{status, filename, chunks_added}"]
    G1 --> I
    D -. exception .-> Z["HTTP 500<br/>Ingestion failed"]
```

## `POST /query` — answer a question

```mermaid
flowchart TD
    A["Client<br/>POST /query {question}"] --> B{"question<br/>non-empty?"}
    B -- no --> B1["HTTP 400<br/>question must not be empty"]
    B -- yes --> C["engine.answer(question)"]

    subgraph answer ["RAGEngine.answer"]
        C --> D["retrieve<br/>(threadpool)"]
        subgraph retrieval ["Retrieval chain"]
            D --> E["Dense + sparse<br/>Pinecone query"]
            E --> F["Cohere rerank"]
            F --> G["Top-K docs"]
        end
        G --> H{"any docs?"}
        H -- no --> H1["answer = 'no indexed<br/>documents' · empty lists"]
        H -- yes --> I["Format context<br/>(citation-tagged blocks)"]
        I --> J["generation_chain.ainvoke<br/>Gemini → structured RAGAnswer"]
        J --> K["Build citations<br/>from doc metadata"]
    end

    K --> L["HTTP 200<br/>{answer, insights,<br/>next_questions, citations}"]
    H1 --> L
    C -. exception .-> Z["HTTP 500<br/>Query failed"]
```

### Structured generation output (`RAGAnswer`)

```mermaid
flowchart LR
    R["RAGAnswer"] --> A["answer<br/>direct grounded answer"]
    R --> B["insights[]<br/>implications / next steps"]
    R --> C["next_questions[]<br/>suggested follow-ups"]
```
