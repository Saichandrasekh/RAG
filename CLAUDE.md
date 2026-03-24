# Policy RAG — Development Guide for Claude

## Project Overview
**Policy RAG** is a production-grade Retrieval-Augmented Generation (RAG) system for document Q&A. Users upload documents (PDF, DOCX, TXT) and media files (MP4, MP3, WAV), and the system answers questions by retrieving relevant chunks and generating responses using an LLM. Supports text, tables, and audio/video transcripts. Includes text-to-speech playback of answers.

**Live instance**: Azure Container Apps (auto-deployed from `sai1` branch)

---

## Architecture at a Glance

```
User Browser (Vanilla JS + SSE)
         ↓ HTTP/SSE
   Flask + Gunicorn
    (1 worker, 4 threads)
         ↓
    ┌─────────────┐
    │  ChromaDB   │ ← Vector DB (local SSD, synced to Azure Blob)
    │             │
    │ Retrieval:  │
    │  • Query expansion (LLM-based)
    │  • Multi-query semantic search (3× ChromaDB calls)
    │  • BM25 + RRF hybrid reranking
    │  • Adjacent chunk expansion (N-1, N, N+1)
    │  • Document-order sorting
    └─────────────┘
         ↓
    Groq API (llama-3.1-8b-instant)
    SSE streaming → Browser
```

---

## Key Design Decisions (Why?)

### Threading & Concurrency
- **1 Gunicorn worker + 4 threads** (not multi-process)
  - **Why**: ChromaDB is not fork-safe; multi-process would require separate collection instances
  - **Implication**: All threads share the same ChromaDB collection object (thread-safe)

### Chunking Strategy
- **3 sentences per chunk** with **1 sentence overlap**
  - **Why**: Smaller chunks = more focused embeddings = better retrieval accuracy
  - **Trade-off**: More chunks → slightly higher storage; negligible since we use BatchUpsert

### Query Expansion
- **LLM-based** (not rule-based)
  - **Why**: Handles terminology variants ("mobile" vs "phone", "contact" vs "details")
  - **How**: Fast non-streaming call with max_tokens=60; graceful fallback if LLM fails

### Multi-Query Retrieval
- **Run ChromaDB 3 times** (original + 2 expansions) → deduplicate by chunk_id
  - **Why**: Robustness; if one phrasing fails, others may succeed
  - **Deduplication**: Preserves best similarity score (min distance) per chunk

### Hybrid Search (BM25 + RRF)
- **Semantic ranking** (by ChromaDB distance) + **BM25 keyword ranking** → fused with RRF
  - **Why**: Exact-match queries (phone numbers, names, dates) need keyword strength
  - **RRF k=60**: Industry standard from Cormack et al. (2009)
  - **Formula**: `1/(rank_semantic + 60) + 1/(rank_bm25 + 60)`
  - **BM25 corpus**: Only the retrieved candidates (40-100 chunks), not the entire collection
    - Makes IDF discriminative for the query context (better than corpus-wide IDF)

### Adjacent Chunk Expansion
- **Fetch N-1 and N+1** for each top-k chunk → merge into ~9-sentence context
  - **Why**: LLM sees coherent document flow; better answers for boundary-spanning questions
  - **Fallback**: If N-1 or N+1 don't exist, skip gracefully

### Document-Order Sorting
- **Group chunks by source file** → sort by chunk index within each source
  - **Why**: Chunks from the same document appear sequentially; LLM reads naturally

### ChromaDB Persistence
- **Local SSD** for fast queries + **Blob Storage sync** for persistence
  - **Why**: Fast local reads; index survives container restarts without re-indexing
  - **How**: Download on startup, upload after ingest/delete (background thread)

### Per-File Source Check
- **`collection.get(where={"source": filename})`** instead of bulk `collection.get()`
  - **Why**: Avoids "SQLite too many SQL variables" error on large collections
  - **Trade-off**: Slightly slower than bulk get; necessary for reliability

### SSE Streaming
- **Token-by-token streaming** for LLM responses
  - **Why**: Perceived latency reduced; words appear instantly instead of 15s wait
  - **Implementation**: `response.iter_lines()` → `yield f"data: {chunk}\n\n"`

### Min Replicas = 1
- **Always-on container** (never scale to zero)
  - **Why**: Model loading takes 20-30s from cold start; poor UX for first query
  - **Cost**: Minimal (1 vCPU always running)

---

## Key Files & Their Responsibilities

| File | Purpose | Modifications |
|---|---|---|
| `app.py` | Main Flask app — routes, retrieval pipeline, LLM streaming, Whisper transcription, cleanup | Retrieval logic, metadata propagation, multi-modal prompt building, audio/video transcription, batch delete |
| `ingest.py` | Document ingestion — parsing, chunking, ChromaDB upsert | Type-aware chunking (text, table, transcript), type-specific chunk IDs |
| `utils.py` | File parsers, table extractors, audio processing, transcript chunking | Multi-modal extraction pipeline |
| `blob_sync.py` | Azure Blob Storage sync — ChromaDB index download/upload | Index persistence across deployments |
| `requirements.txt` | Python dependencies | `rank-bm25`, `moviepy` |
| `templates/index.html` | Jinja2 frontend template | Play button (TTS), type-specific source icons, audio/video upload |
| `TECH_STACK.md` | Full documentation (keep in sync!) | Updated with current features |

---

## Code Conventions & Patterns

### Function Organization in `app.py`
1. **Imports & init** (top ~50 lines)
2. **Utility functions**: `tokenize()`, `rerank_and_filter_chunks()`, `expand_query()`, `sort_chunks_by_document_order()`, `fetch_adjacent_chunks()`
3. **Audio/Video**: `transcribe_file()` (Groq Whisper API)
4. **Core retrieval**: `retrieve_chunks()`, `api_search_stream()`, `api_delete_file()`
5. **Flask routes**: `/`, `/api/search_stream`, `/api/delete_file`, etc.

### Error Handling
- **No silent failures**: Always log errors with context
- **Graceful degradation**: If BM25 unavailable → use semantic-only ranking
- **Chunk fetching**: If N-1 or N+1 missing → skip, don't crash

### Variable Naming
- `chunk_id` — unique identifier (format: `{filename}_chunk_{index}`, `{filename}_table_{page}_{idx}`, `{filename}_transcript_{idx}`)
- `distance` — L2 distance from ChromaDB (lower = more similar)
- `source` — filename/path of the chunk's source document
- `top_k` — number of chunks to return to LLM
- `type` — chunk content type: `"text"`, `"table"`, `"transcript"` (default: `"text"` for backward compat)

---

## Retrieval Pipeline (Step by Step)

```
1. User asks: "Give me SAI's mobile number"
   ↓
2. expand_query() → ["Give me SAI's mobile number", "What is SAI's phone?", "SAI contact details"]
   ↓
3. retrieve_chunks(query, top_k=12)
   • For each of 3 queries: run ChromaDB with initial_k=72 (top_k*6)
   • Collect results, deduplicate by chunk_id (keep best distance)
   • Build text_to_id map for later matching
   ↓
4. rerank_and_filter_chunks()
   • Hard pre-filter: drop chunks with distance > 1.5
   • Semantic ranking: sort by distance → assign ranks 0,1,2...
   • BM25 ranking: tokenize query, build BM25Okapi corpus on candidates, get scores → assign ranks
   • RRF fusion: combine scores using 1/(rank_sem + 60) + 1/(rank_bm25 + 60)
   • Deduplicate by normalized text (case-insensitive, whitespace-normalized)
   • Return top 12 dicts: {source, distance, text, page, chunk_id}
   ↓
5. sort_chunks_by_document_order()
   • Group by source filename
   • Sort within each source by chunk index (extracted from chunk_id suffix)
   ↓
6. fetch_adjacent_chunks()
   • For each chunk in top 12, fetch N-1 and N+1 from same source
   • Merge: [N-1 text] + [N text] + [N+1 text] (space-separated)
   • Return merged chunks (strip chunk_id from text)
   ↓
7. Build prompt: "Context:\n{merged_chunks}\n\nQuestion: {user_query}"
   ↓
8. Stream response from Groq API (token by token via SSE)
```

---

## Development Workflow

### Local Setup
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
python app.py  # Runs on localhost:5000
```

### Testing Query Expansion
```python
from app import expand_query
result = expand_query("give me phone number")
print(result)  # Should return [original, alt1, alt2]
```

### Testing BM25 Ranking
1. Upload a document with a phone number
2. Query: "phone number" (exact match)
3. Check logs for `[RERANK]` messages
4. Phone number chunk should rank high despite low semantic similarity

### Testing Adjacent Chunk Expansion
1. Query a word that appears at chunk boundary
2. Verify merged result includes context from N-1 and N+1

### Adding a New Document Format
1. Add a `stream_xyz()` function to `utils.py`
2. Add case in `yield_file_chunks()` to detect and route the new format
3. Test chunking with `chunk_text_stream()`
4. No changes to retrieval pipeline needed

---

## Known Gotchas & Gotchas

### ChromaDB SQLite Limits
- **Issue**: "SQLite too many SQL variables" on large collections
- **Cause**: `collection.get(ids=[...])` with huge ID list
- **Fix**: Use per-file source check instead: `collection.get(where={"source": filename})`

### Cold Start Latency
- **Issue**: First query after deploy takes 20-30s
- **Cause**: Model loading (SentenceTransformers → ChromaDB)
- **Why min replicas=1**: Avoids this; always warm
- **Trade-off**: Minimal cost (1 vCPU always on)

### BM25 Not Installed
- **Issue**: `ImportError: No module named 'rank_bm25'`
- **Fix**: `pip install rank-bm25`
- **Fallback**: If not installed, uses semantic-only ranking (no error, graceful)

### Long Ingestion Time
- **Current**: Batch upsert (100 chunks per insert) → SQLite overhead
- **Mitigation**: Acceptable trade-off; async ingest doesn't block HTTP

### Large File Deletion
- **Issue**: `collection.get(where={"source": filename})` crashes on files with many chunks
- **Fix**: Delete in batches of 500 IDs per loop iteration
- **Endpoint**: `/api/cleanup_index` — removes orphaned index entries for deleted files (batched)

### Duplicate Chunks in Results
- **Cause**: Same content embedded in multiple pages/documents
- **Fix**: Deduplication by normalized text (case-insensitive, whitespace-normalized)

### Adjacent Chunks Causing Redundancy
- **Issue**: N-1, N, N+1 might be 90% overlapping
- **Design choice**: This is intentional! More context is better for LLM
- **Mitigation**: LLM can ignore redundancy; focuses on relevant parts

---

## Deployment & CI/CD

### GitHub Actions
- **Trigger**: Push to `sai1` branch
- **Pipeline**: Build Docker image → push to ACR → deploy to Container Apps
- **Environment**: `.env` file (secrets via GitHub secrets, not in repo)

### Azure Container Apps Config
```
Min replicas: 1
Max replicas: 10
CPU: 2 cores
Memory: 4 GB
Cooldown: 300s
Polling interval: 30s
Timeout: 600s (matches Gunicorn timeout for large file ingest)
```

### Manual Deploy
```bash
docker build -t myregistry.azurecr.io/policyrag:latest .
docker push myregistry.azurecr.io/policyrag:latest
az containerapp update -n policyrag -g mygroup --image myregistry.azurecr.io/policyrag:latest
```

---

## Environment Variables (Required)

```
XAI_API_KEY=gsk_...
XAI_BASE_URL=https://api.groq.com/openai/v1
XAI_MODEL=llama-3.1-8b-instant
RETRIEVAL_TOP_K=12  # Chunks to pass to LLM
RETRIEVAL_MAX_DISTANCE=1.5  # Max L2 distance filter
AZURE_STORAGE_CONNECTION_STRING=DefaultEndpointsProtocol=...
AZURE_BLOB_CONTAINER=policyrag-index
FLASK_SECRET_KEY=your-secret-here
```

---

## Future Improvements (Roadmap)

1. **Conversational Memory**: Keep chat history, use for context
2. **User Authentication**: Isolate conversations per user
3. **Analytics**: Track queries, popular topics, retrieval performance
4. **API-First Architecture**: Separate API backend from frontend
5. **Cost Optimization**: Batch queries, cache embeddings
6. ~~**Multi-Modal RAG**: Support images, tables, videos (not just text)~~ **DONE** — Tables (PDF/DOCX), audio/video (Whisper transcription)
9. ~~**Text-to-Speech**: Play button for LLM answers~~ **DONE** — Browser Web Speech API with Indian English voice (en-IN)
10. ~~**Batch Delete & Cleanup**: Fix SQLite limits on large file deletion~~ **DONE** — Batch delete (500 IDs) + orphan cleanup endpoint
11. ~~**Docker Layer Caching**: Faster CI/CD deployments~~ **DONE** — Buildx with registry-based cache
7. **Reranker Model**: Use a dedicated reranker (ColBERT, MonoBERT) instead of BM25
8. **Semantic Caching**: Cache embeddings for common queries

---

## Commit Message Style

- **Feat**: `feat: implement [feature name]` (new functionality)
- **Fix**: `fix: [issue]` (bug fixes)
- **Docs**: `docs: [what changed]` (documentation updates)
- **Refactor**: `refactor: [what changed]` (code reorganization, no behavior change)
- **Test**: `test: [what tested]` (test additions)
- **Chore**: `chore: [what]` (dependency updates, config changes)

**Examples**:
- `feat: implement multi-document context window with query expansion`
- `fix: replace bulk collection.get() with per-file check to avoid SQLite too many SQL variables error`
- `docs: update tech stack with hybrid search features`

---

## Questions for Claude (You!)

If you're debugging or making changes, ask yourself:

1. **Will this change affect retrieval quality?** → Test with specific queries
2. **Will this break ChromaDB multi-threading?** → Use thread-safe operations only
3. **Will this increase latency?** → Benchmark (should be <100ms added)
4. **Is BM25 available?** → Check `_BM25_AVAILABLE` guard
5. **Does the index need re-indexing?** → Only if chunking strategy changes
6. **Will Azure storage be affected?** → Check `blob_sync.py` flows

---

## Running Tests Locally

```bash
# Test query expansion
python -c "from app import expand_query; print(expand_query('phone number'))"

# Test retrieval with BM25
python -c "from app import retrieve_chunks; chunks = retrieve_chunks('phone', top_k=5); print(chunks)"

# Test HTML rendering
curl http://localhost:5000/

# Test SSE streaming
curl -N http://localhost:5000/api/search_stream \
  -H "Content-Type: application/json" \
  -d '{"question": "what is X?", "top_k": 12}'
```

---

**Last Updated**: 2026-03-24
**Branch**: `sai1` (main development branch)
**Maintainers**: You (and any other collaborators)
