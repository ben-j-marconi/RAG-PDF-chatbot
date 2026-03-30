# Property Valuation Tool — RAG Chatbot

A local-first RAG system for property valuation intelligence. Designed to solve a real problem in real estate: proprietary valuation data trapped in siloed, heterogeneous PDF documents that standard RAG tools can't parse reliably.

**The problem:** A real estate firm needs an evidence-backed Q&A system over internal valuation documents — appraisals, rent rolls, market comps, condition assessments, and underwriting narratives. Standard RAG pipelines fail because they treat PDFs as flat text, destroying table structure, losing section context, and polluting chunks with header/footer noise.

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure environment
cp .env.example .env
# Edit .env → add OPENAI_API_KEY=sk-...

# 3. Ingest documents (downloads ~180MB of models on first run)
python3 ingest.py

# 4. Launch the app
streamlit run app.py
# Opens at http://localhost:8501
```

---

## Environment Variables

### Local / Dev Mode (default)
```bash
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini
```

### Vertex AI Gemini Mode (production)
```bash
LLM_PROVIDER=vertex
GOOGLE_CLOUD_PROJECT=your-gcp-project-id
GOOGLE_CLOUD_LOCATION=us-central1
VERTEX_MODEL=gemini-2.0-flash-001
```
Then run:
```bash
pip install google-cloud-aiplatform
gcloud auth application-default login
```
No code changes required. The generation layer is fully swappable via config.

---

## Ingest Commands

```bash
python3 ingest.py              # Delta ingest: new/changed files only
python3 ingest.py --force      # Full reindex: all documents
python3 ingest.py --status     # Show what's indexed without processing
```

---

## Project Structure

```
RAG_chatbot/
├── data/
│   ├── pdfs/                   # Source PDFs → place documents here
│   ├── registry/               # document_registry.json + bm25_index.pkl
│   └── chroma_db/              # ChromaDB vector store (local persistent)
├── src/
│   ├── ingestion/
│   │   ├── registry.py         # Canonical doc registry (JSON + SHA-256 hashing)
│   │   ├── scanner.py          # Detects new/changed/deleted PDFs
│   │   └── pipeline.py         # Orchestrates the full ingest pipeline
│   ├── parsing/
│   │   ├── detector.py         # Native text vs. scanned PDF detection
│   │   ├── extractor.py        # pdfplumber: extracts text + tables separately
│   │   ├── cleaner.py          # Strips headers/footers, normalizes artifacts
│   │   ├── reconstructor.py    # Rebuilds document structure (headings, sections)
│   │   └── ocr_adapter.py      # OCR interface stub (Document AI in production)
│   ├── chunking/
│   │   └── chunker.py          # Section-aware + table-preserving chunking
│   ├── indexing/
│   │   ├── embedder.py         # Local SentenceTransformer embeddings
│   │   ├── vector_store.py     # ChromaDB wrapper (upsert, query, delete)
│   │   └── lexical_index.py    # BM25Okapi index (pickle-backed)
│   ├── retrieval/
│   │   ├── retriever.py        # Hybrid retriever: vector + BM25 + RRF merge
│   │   └── reranker.py         # Cross-encoder reranker (ms-marco-MiniLM)
│   ├── llm/
│   │   ├── base.py             # Abstract LLMProvider + grounding prompt
│   │   ├── openai_provider.py  # OpenAI (dev mode)
│   │   └── vertex_provider.py  # Vertex AI Gemini (production mode)
│   ├── chat/
│   │   └── engine.py           # Coordinates retrieval → rerank → generate
│   └── observability/
│       └── logger.py           # Structured JSON logging + eval hooks
├── app.py                      # Streamlit UI
├── ingest.py                   # CLI ingestion script
├── config.py                   # Environment config loader
├── requirements.txt
└── .env.example
```

---

## Architecture

```
PDF files in ./data/pdfs/
         │
         ▼
┌─────────────────────────────────────────┐
│  1. PDF Acquisition & Source Federation │
│     - SHA-256 hash-based change detect  │
│     - Canonical doc registry (JSON)     │
│     - Delta ingest (skip unchanged)     │
└─────────────────────┬───────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────┐
│  2. Document Restructuring              │
│     - Detect: native text vs. scanned   │
│     - Extract: text + tables separately │
│     - Clean: strip headers/footers      │
│     - Reconstruct: headings, sections   │
│     → StructuredKnowledgeObject         │
└─────────────────────┬───────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────┐
│  3. Knowledge Indexing                  │
│     - Section-aware chunking            │
│     - Tables preserved as Markdown      │
│     - Metadata: doc_id, page, type,     │
│       section heading                   │
│     - ChromaDB (vector)                 │
│     - BM25Okapi (lexical)               │
└─────────────────────┬───────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────┐
│  4. Retrieval & Grounding               │
│     - Vector search (top-20)            │
│     - BM25 search (top-20)              │
│     - Reciprocal Rank Fusion merge      │
│     - Cross-encoder reranking (top-5)   │
│     → grounded evidence context         │
└─────────────────────┬───────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────┐
│  5. Response Layer (Chatbot)            │
│     - Pluggable LLM provider            │
│       · Dev:  OpenAI gpt-4o-mini        │
│       · Prod: Vertex AI Gemini          │
│     - Citations: doc + page + section   │
│     - Confidence indicator              │
│     - Retrieval debug view              │
└─────────────────────┬───────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────┐
│  6. Observability & Trust               │
│     - Structured JSON logs              │
│     - Extraction quality checks         │
│     - Retrieval relevance logging       │
│     - Groundedness heuristic check      │
└─────────────────────────────────────────┘
```

---

## Why This Architecture

### PDF != Flat Text
Standard loaders call `extract_text()` and stop. Our pipeline:
- Extracts tables **separately** from text using pdfplumber's spatial analysis
- Strips header/footer artifacts that would pollute every chunk
- Reconstructs section headings as metadata (not lost in flat text)
- Preserves table structure as Markdown (not collapsed into a single string)

### Hybrid Retrieval
Vector search alone misses exact terms: "DSCR 1.31x", "$7,880,000", unit numbers. BM25 alone misses semantic paraphrase: "net operating income" vs "NOI". Combining both via Reciprocal Rank Fusion covers both failure modes without requiring score normalization.

### Cross-Encoder Reranking
The bi-encoder (embedding similarity) encodes query and document independently. The cross-encoder reads them together, capturing query-document interaction that dot-product similarity misses. For financial tables with specific numbers, this matters.

### Provenance and Grounded Answers
Every chunk carries `document_id`, `source_filename`, `page_num`, `chunk_type`, and `section_heading`. Every answer cites specific chunks. The grounding prompt explicitly prohibits using external knowledge.

### Local-First, Cloud-Ready
- Dev: `LLM_PROVIDER=openai` — runs on any laptop
- Prod: `LLM_PROVIDER=vertex` — one env var, zero code changes
- The same swap applies to embeddings (→ text-embedding-004) and vector store (→ Vertex AI Vector Search) in a full GCP deployment

---

## Design Tradeoffs

| Decision | Choice | Alternative | Why |
|---|---|---|---|
| PDF parser | pdfplumber | PyPDF2, PyMuPDF | Best table extraction for native text PDFs |
| Chunking | Section-aware | Fixed-size | Preserves table semantics; headings as metadata |
| Retrieval | Hybrid (RRF) | Vector only | Handles exact terms + semantic paraphrase |
| Reranker | Cross-encoder | Score fusion | Joint query-doc scoring; better precision |
| Vector store | ChromaDB | FAISS, Weaviate | Local persistent; zero infrastructure |
| LLM (dev) | OpenAI gpt-4o-mini | Local LLM | Reliable, fast, cheap; easy to explain |
| LLM (prod) | Vertex AI Gemini | OpenAI in prod | GCP-native; integrates with Vertex AI RAG Engine |

---

## Known Limitations (MVP scope)

- OCR not implemented: scanned PDFs extract partial text only (Document AI in production)
- No authentication layer: ACL metadata tracked in registry but not enforced at retrieval
- BM25 index held in memory + pickle: not suitable for large corpora (Elasticsearch in production)
- No query expansion or HyDE: query is used as-is for retrieval
- Evaluation is heuristic only: no golden eval set or automated groundedness metrics

---

## Frequently Asked Questions

**Why does standard RAG fail on heterogeneous PDFs?**
Three reasons: (1) Fixed-size chunking splits tables mid-row, destroying the row-column relationship that gives numbers their meaning. (2) Headers and footers repeat on every page and pollute every chunk with noise tokens. (3) Flat text extraction loses section structure, so retrieval can't distinguish "income approach" data from "risk factors" in the same document.

**Why RRF instead of a linear combination of vector + BM25 scores?**
Linear combination requires normalizing across fundamentally different score distributions — cosine similarity (0 to 1) vs BM25 TF-IDF scores (unbounded). RRF works in rank space: it adds `1 / (k + rank)` contributions from each ranked list. Documents ranked highly in both lists get boosted naturally, with no normalization required. This is the same approach used by Elasticsearch 8.x and Vertex AI RAG Engine.

**How does the cross-encoder improve over embedding similarity for ranking?**
The bi-encoder encodes query and document independently — the relevance score is a dot product of two independent vectors and can't capture query-document interaction. The cross-encoder concatenates query and document into a single input, allowing it to understand that "DSCR" in the query matches "debt service coverage ratio 1.31x" in the document. The tradeoff is speed: bi-encoder is O(1) at query time; cross-encoder is O(n) in the candidate set — for 30 candidates, this is milliseconds.

**How would you move this to production on Google Cloud?**
The generation layer is already provider-agnostic — set `LLM_PROVIDER=vertex`. Then swap three components: (1) embedding model from all-MiniLM to Vertex AI text-embedding-004, (2) ChromaDB to Vertex AI Vector Search for managed billion-scale ANN, (3) BM25 to Vertex AI Search or Elasticsearch on GKE. Add IAM to enforce the ACL metadata the registry already tracks, and route logs to Cloud Logging. Application code doesn't change — only config adapters.

**How does the system prevent hallucination?**
Three mechanisms: (1) The system prompt explicitly prohibits using external knowledge and requires every financial figure to cite its source document. (2) The groundedness eval hook compares numbers in the answer against numbers in retrieved evidence — any figure not present in context is flagged in logs. (3) The confidence field reflects how directly the retrieved evidence addresses the query, surfaced in the UI so users know when evidence is weak.
