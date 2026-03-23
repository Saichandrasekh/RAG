from flask import Flask, render_template, request, jsonify, Response, stream_with_context
import chromadb
from chromadb.utils import embedding_functions
import os
import re
import json
import shutil
import threading
import logging
from dotenv import load_dotenv
from openai import OpenAI
from werkzeug.utils import secure_filename
from ingest import ingest_new_files, DATA_DIR
from utils import is_audio_video, extract_audio, split_audio_for_whisper, chunk_transcript
from blob_sync import download_index, upload_index, download_images, upload_images

# ── BM25 ranking (optional dependency for hybrid search) ────────────────────
try:
    from rank_bm25 import BM25Okapi
    _BM25_AVAILABLE = True
except ImportError:
    BM25Okapi = None
    _BM25_AVAILABLE = False

# ── Logging setup ────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Suppress noisy Azure SDK HTTP logs
logging.getLogger("azure.core.pipeline.policies.http_logging_policy").setLevel(logging.WARNING)
logging.getLogger("azure.storage").setLevel(logging.WARNING)

# ── App init ─────────────────────────────────────────────────────────────────
load_dotenv()
openai_client = OpenAI(
    api_key=os.getenv("XAI_API_KEY"),
    base_url=os.getenv("XAI_BASE_URL", "https://api.x.ai/v1"),
)

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "dev-secret-key")
ALLOWED_EXTENSIONS = {".txt", ".pdf", ".docx", ".mp4", ".mp3", ".wav", ".m4a", ".webm"}

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "how", "i",
    "in", "is", "it", "of", "on", "or", "that", "the", "this", "to", "was", "what",
    "when", "where", "which", "who", "why", "with", "you", "your"
}

# ── Blob Storage: download index on startup ───────────────────────────────────
INDEX_DIR = "index/chroma_db"
if os.getenv("AZURE_STORAGE_CONNECTION_STRING"):
    logger.info("[BLOB] Downloading ChromaDB index from Blob Storage...")
    success = download_index(INDEX_DIR)
    if success:
        logger.info("[BLOB] Index restored from Blob Storage.")
    else:
        logger.info("[BLOB] No index in Blob — will start fresh and build on first ingest.")
    # Also download extracted images
    download_images()
else:
    logger.info("[BLOB] AZURE_STORAGE_CONNECTION_STRING not set — skipping Blob sync (local mode).")

# ── ChromaDB init ─────────────────────────────────────────────────────────────
logger.info("Initializing ChromaDB...")
os.makedirs(INDEX_DIR, exist_ok=True)
chroma_client = chromadb.PersistentClient(path=INDEX_DIR)
sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
collection = chroma_client.get_or_create_collection(
    name="knowledge_base",
    embedding_function=sentence_transformer_ef
)
logger.info(f"ChromaDB ready. Total chunks in index: {collection.count()}")

# Track background ingest jobs: filename -> status
ingest_status = {}


# ── Audio/Video transcription ─────────────────────────────────────────────────
def transcribe_file(file_path, filename):
    """Transcribe an audio/video file using Groq Whisper API.
    Returns list of transcript chunks as (text, page, metadata_dict) tuples."""
    logger.info(f"[TRANSCRIBE] Starting transcription for: {filename}")

    # Extract audio (converts to WAV if needed)
    wav_path = extract_audio(file_path)

    # Split if >25MB (Groq Whisper limit)
    segments_paths = split_audio_for_whisper(wav_path)

    all_segments = []
    time_offset = 0.0

    for seg_path in segments_paths:
        try:
            with open(seg_path, "rb") as audio_file:
                transcript = openai_client.audio.transcriptions.create(
                    model="whisper-large-v3",
                    file=audio_file,
                    response_format="verbose_json",
                )

            # Extract segments with timestamps
            if hasattr(transcript, "segments") and transcript.segments:
                for seg in transcript.segments:
                    all_segments.append({
                        "text": seg.get("text", seg.text if hasattr(seg, "text") else ""),
                        "start": (seg.get("start", 0) if isinstance(seg, dict) else getattr(seg, "start", 0)) + time_offset,
                        "end": (seg.get("end", 0) if isinstance(seg, dict) else getattr(seg, "end", 0)) + time_offset,
                    })
            elif hasattr(transcript, "text") and transcript.text:
                # Fallback: no segments, just full text
                all_segments.append({
                    "text": transcript.text,
                    "start": time_offset,
                    "end": time_offset + 600,
                })

            # Update time offset for next segment
            if all_segments:
                time_offset = all_segments[-1]["end"]

        except Exception as e:
            logger.error(f"[TRANSCRIBE] Whisper API failed for segment: {e}")
            continue

    # Clean up temp WAV files
    if wav_path != file_path and os.path.exists(wav_path):
        try:
            os.remove(wav_path)
        except OSError:
            pass
    for seg_path in segments_paths:
        if seg_path != wav_path and seg_path != file_path and os.path.exists(seg_path):
            try:
                os.remove(seg_path)
            except OSError:
                pass

    # Chunk the transcript
    chunks = []
    for idx, (text, timestamp) in enumerate(chunk_transcript(all_segments)):
        chunks.append((text, None, {"type": "transcript", "timestamp": timestamp}))

    logger.info(f"[TRANSCRIBE] Generated {len(chunks)} transcript chunks for: {filename}")
    return chunks


# ── Ingest background thread ──────────────────────────────────────────────────
def run_ingest_background(filename):
    logger.info(f"[INGEST] Starting background ingest for: {filename}")
    ingest_status[filename] = "processing"
    try:
        file_path = os.path.join(DATA_DIR, filename)

        # Handle audio/video files: transcribe first, then ingest transcript chunks
        if is_audio_video(file_path):
            transcript_chunks = transcribe_file(file_path, filename)
            if transcript_chunks:
                # Ingest transcript chunks directly
                from ingest import BATCH_SIZE, log as ingest_log
                batch_docs, batch_metas, batch_ids = [], [], []
                for idx, (text, page, meta) in enumerate(transcript_chunks):
                    batch_docs.append(text)
                    batch_metas.append({
                        "source": filename,
                        "page": page if page is not None else -1,
                        "type": meta.get("type", "transcript"),
                        "timestamp": meta.get("timestamp", ""),
                    })
                    batch_ids.append(f"{filename}_transcript_{idx}")

                    if len(batch_docs) >= BATCH_SIZE:
                        collection.upsert(documents=batch_docs, metadatas=batch_metas, ids=batch_ids)
                        batch_docs, batch_metas, batch_ids = [], [], []

                if batch_docs:
                    collection.upsert(documents=batch_docs, metadatas=batch_metas, ids=batch_ids)

                logger.info(f"[INGEST] Transcribed and indexed {len(transcript_chunks)} chunks for: {filename}")
            else:
                logger.warning(f"[INGEST] No transcript generated for: {filename}")
        else:
            # Standard document ingestion
            ingest_new_files(collection=collection)

        ingest_status[filename] = "done"
        logger.info(f"[INGEST] Completed successfully: {filename} | Total chunks: {collection.count()}")
        # Upload updated index to Blob Storage so it persists across deployments
        if os.getenv("AZURE_STORAGE_CONNECTION_STRING"):
            logger.info("[BLOB] Uploading updated index to Blob Storage...")
            upload_index(INDEX_DIR)
            upload_images()
    except Exception as e:
        ingest_status[filename] = f"error: {e}"
        logger.error(f"[INGEST] Failed for {filename}: {e}", exc_info=True)


def is_allowed_file(filename):
    return os.path.splitext(filename.lower())[1] in ALLOWED_EXTENSIONS


def delete_file_and_index(filename):
    logger.info(f"[DELETE] Removing file and index for: {filename}")
    file_path = os.path.join(DATA_DIR, filename)
    if os.path.exists(file_path):
        os.remove(file_path)
        logger.info(f"[DELETE] File removed: {file_path}")
    # Clean up extracted images
    filename_stem = os.path.splitext(filename)[0]
    images_dir = os.path.join("static", "images", filename_stem)
    if os.path.isdir(images_dir):
        shutil.rmtree(images_dir, ignore_errors=True)
        logger.info(f"[DELETE] Removed images dir: {images_dir}")
    results = collection.get(where={"source": filename})
    if results and results["ids"]:
        collection.delete(ids=results["ids"])
        logger.info(f"[DELETE] Removed {len(results['ids'])} chunks from index for: {filename}")
        if os.getenv("AZURE_STORAGE_CONNECTION_STRING"):
            threading.Thread(target=upload_index, args=(INDEX_DIR,), daemon=True).start()


# ── Retrieval ─────────────────────────────────────────────────────────────────
def tokenize(text):
    return re.findall(r"[a-z0-9]+", text.lower())


def rerank_and_filter_chunks(raw_chunks, query, top_k, max_distance):
    # ── Step 1: Hard distance pre-filter ────────────────────────────────────
    candidates = [c for c in raw_chunks if c["distance"] <= max_distance]

    # Fallback: if pre-filter removes everything, return top_k by distance
    if not candidates:
        raw_by_distance = sorted(raw_chunks, key=lambda x: x["distance"])
        return [
            {**i, "score": round(i["distance"], 3)}
            for i in raw_by_distance[:top_k]
        ]

    # ── Step 2: Semantic ranking ───────────────────────────────────────────
    candidates_sorted_semantic = sorted(candidates, key=lambda x: x["distance"])
    semantic_rank = {c["text"]: rank for rank, c in enumerate(candidates_sorted_semantic)}

    # ── Step 3: BM25 ranking + RRF fusion ──────────────────────────────────
    RRF_K = 60  # Standard constant from Cormack et al., 2009

    if _BM25_AVAILABLE and len(candidates) > 0:
        # Tokenize corpus and query using the same tokenizer
        tokenized_corpus = [
            [tok for tok in tokenize(c["text"]) if tok not in STOPWORDS]
            for c in candidates
        ]
        tokenized_query = [tok for tok in tokenize(query) if tok not in STOPWORDS]

        # Build BM25 index and score candidates
        bm25 = BM25Okapi(tokenized_corpus)
        bm25_scores = bm25.get_scores(tokenized_query)

        # Rank candidates by BM25 score (higher = better)
        bm25_ranked = sorted(
            enumerate(candidates),
            key=lambda x: bm25_scores[x[0]],
            reverse=True
        )
        bm25_rank = {c["text"]: rank for rank, (_, c) in enumerate(bm25_ranked)}

        # ── RRF fusion: combine semantic and BM25 ranks ─────────────────────
        # RRF score = 1/(rank_semantic + k) + 1/(rank_bm25 + k)
        rrf_scored = []
        for c in candidates:
            r_sem = semantic_rank[c["text"]]
            r_bm25 = bm25_rank[c["text"]]
            rrf_score = 1.0 / (r_sem + RRF_K) + 1.0 / (r_bm25 + RRF_K)
            rrf_scored.append((rrf_score, c))

        rrf_scored.sort(key=lambda x: x[0], reverse=True)
        ranked = [c for _, c in rrf_scored]

    else:
        # Graceful degradation: BM25 unavailable or no candidates
        if not _BM25_AVAILABLE:
            logger.warning("[RERANK] rank-bm25 not available; falling back to semantic-only ranking.")
        ranked = candidates_sorted_semantic

    # ── Step 4: Deduplicate by normalized text, return top_k ────────────────
    deduped = []
    seen_texts = set()
    for c in ranked:
        key = " ".join(c["text"].split()).lower()
        if key in seen_texts:
            continue
        seen_texts.add(key)
        deduped.append({
            **c,
            "score": round(c["distance"], 3),
        })
        if len(deduped) >= top_k:
            break

    return deduped


def expand_query(query: str) -> list:
    """
    Generate 2 alternative phrasings of the query using the LLM.
    Returns a list that always starts with the original query.
    Falls back to [query] on any error.
    """
    system_msg = "You are a search query expansion assistant."
    user_msg = (
        f"Generate 2 alternative phrasings of this search query. "
        f"Return ONLY a JSON array of strings, no explanation.\n"
        f"Query: {query}\n"
        f"Example output: [\"alt phrasing 1\", \"alt phrasing 2\"]"
    )
    try:
        response = openai_client.chat.completions.create(
            model=os.getenv("XAI_MODEL", "grok-3-latest"),
            temperature=0,
            stream=False,
            max_tokens=60,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
        )
        raw = response.choices[0].message.content.strip()
        # Strip markdown code fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        alternatives = json.loads(raw)
        if not isinstance(alternatives, list):
            raise ValueError("LLM did not return a list")
        # Prepend the original query so it always leads
        queries = [query] + [q for q in alternatives if isinstance(q, str) and q.strip()]
        logger.info(f"[EXPAND] Expanded '{query}' → {queries}")
        return queries
    except Exception as e:
        logger.warning(f"[EXPAND] Query expansion failed, using original: {e}")
        return [query]


def sort_chunks_by_document_order(chunks: list) -> list:
    """
    Group chunks by source, then within each source sort by the numeric
    suffix of the chunk_id (document order). Sources are ordered by the
    rank of their first-appearing chunk (best relevance first).
    """
    source_order = []
    source_chunks = {}
    for c in chunks:
        src = c["source"]
        if src not in source_chunks:
            source_chunks[src] = []
            source_order.append(src)
        source_chunks[src].append(c)

    result = []
    for src in source_order:
        group = source_chunks[src]
        def sort_key(c):
            cid = c.get("chunk_id", "")
            if cid:
                parts = cid.rsplit("_chunk_", 1)
                if len(parts) == 2:
                    try:
                        return int(parts[1])
                    except ValueError:
                        pass
            return 0
        group.sort(key=sort_key)
        result.extend(group)

    return result


def fetch_adjacent_chunks(chunks: list) -> list:
    """
    For each chunk in chunks, attempt to fetch the previous and next
    chunk from the same source file using their IDs. Merges them into
    a single expanded-text entry. Returns a new list of chunks.
    """
    expanded = []
    for chunk in chunks:
        # Tables and images are self-contained — skip adjacent expansion
        chunk_type = chunk.get("type", "text")
        if chunk_type in ("table", "image"):
            expanded.append(chunk)
            continue

        source = chunk["source"]
        chunk_id = chunk.get("chunk_id")
        if not chunk_id:
            expanded.append(chunk)
            continue

        try:
            parts = chunk_id.rsplit("_chunk_", 1)
            if len(parts) != 2:
                expanded.append(chunk)
                continue
            filename, idx_str = parts
            chunk_idx = int(idx_str)
        except (ValueError, IndexError):
            expanded.append(chunk)
            continue

        # Build the list of IDs to fetch: prev, current, next
        ids_to_fetch = []
        if chunk_idx > 0:
            ids_to_fetch.append(f"{filename}_chunk_{chunk_idx - 1}")
        ids_to_fetch.append(chunk_id)
        ids_to_fetch.append(f"{filename}_chunk_{chunk_idx + 1}")

        try:
            result = collection.get(
                ids=ids_to_fetch,
                include=["documents", "metadatas"]
            )
        except Exception as e:
            logger.warning(f"[EXPAND] collection.get failed for {chunk_id}: {e}")
            expanded.append(chunk)
            continue

        if not result or not result["ids"]:
            expanded.append(chunk)
            continue

        # Build a lookup from ID → (text, page)
        fetched = {}
        for i, fid in enumerate(result["ids"]):
            fetched[fid] = {
                "text": result["documents"][i],
                "page": result["metadatas"][i].get("page", -1),
            }

        # Merge text: prev → current → next
        merged_texts = []
        for fid in ids_to_fetch:
            if fid in fetched:
                merged_texts.append(fetched[fid]["text"].strip())

        merged_text = " ".join(merged_texts)

        # Use the page of the current (center) chunk
        center_page = fetched.get(chunk_id, {}).get("page", chunk.get("page", -1))

        expanded.append({
            **chunk,
            "text": merged_text,
            "page": center_page,
        })

    return expanded


def retrieve_chunks(query, top_k=None):
    if top_k is None:
        top_k = int(os.getenv("RETRIEVAL_TOP_K", "8"))
    max_distance = float(os.getenv("RETRIEVAL_MAX_DISTANCE", "1.4"))

    total_docs = collection.count()
    logger.info(f"[SEARCH] Query: '{query}' | Total indexed chunks: {total_docs}")

    if total_docs == 0:
        logger.warning("[SEARCH] No documents in index.")
        return [], top_k, max_distance

    # Multi-query expansion
    queries = expand_query(query)
    initial_k = min(max(top_k * 6, 40), total_docs)

    # Multi-query retrieval with deduplication by chunk_id (keep best distance)
    seen_ids = {}
    for q in queries:
        results = collection.query(query_texts=[q], n_results=initial_k)
        if not (results and results['documents'] and results['documents'][0]):
            continue
        for i in range(len(results['documents'][0])):
            chunk_id = results['ids'][0][i]
            dist = float(results['distances'][0][i])
            # Keep the best (lowest) distance across all sub-queries
            if chunk_id in seen_ids and seen_ids[chunk_id]["distance"] <= dist:
                continue
            meta = results['metadatas'][0][i]
            seen_ids[chunk_id] = {
                "source": meta.get("source", "Unknown"),
                "distance": dist,
                "text": results['documents'][0][i],
                "page": meta.get("page", -1),
                "chunk_id": chunk_id,
                "type": meta.get("type", "text"),
                "image_path": meta.get("image_path"),
                "timestamp": meta.get("timestamp"),
            }

    raw_chunks = list(seen_ids.values())

    # Build text_to_id map before reranking (which reconstructs dicts)
    text_to_id = {c["text"]: c["chunk_id"] for c in raw_chunks}

    chunks = rerank_and_filter_chunks(raw_chunks, query, top_k=top_k, max_distance=max_distance)

    # Re-attach chunk_id after reranking
    for c in chunks:
        c["chunk_id"] = text_to_id.get(c["text"])

    logger.info(f"[SEARCH] Returning {len(chunks)} chunks for query: '{query}'")
    return chunks, top_k, max_distance


def build_prompt(chunks, query):
    # Group chunks by source for cleaner context
    sources_seen = {}
    for c in chunks:
        src = c["source"]
        if src not in sources_seen:
            sources_seen[src] = []
        chunk_type = c.get("type", "text")
        text = c["text"].strip()
        if chunk_type == "table":
            page = c.get("page", -1)
            label = f"page {page}" if page >= 0 else "unknown page"
            text = f"[Table from {label}]\n{text}"
        elif chunk_type == "image":
            page = c.get("page", -1)
            label = f"page {page}" if page >= 0 else "unknown page"
            text = f"[Image from {label}]\n{text}"
        elif chunk_type == "transcript":
            ts = c.get("timestamp", "")
            text = f"[Transcript {ts}]\n{text}" if ts else text
        sources_seen[src].append(text)

    context_parts = []
    for src, texts in sources_seen.items():
        context_parts.append(f"[From: {src}]\n" + "\n\n".join(texts))
    context = "\n\n---\n\n".join(context_parts)

    return f"""You are a helpful Document Retrieval Assistant. Answer the user's question using ONLY the excerpts provided below.

Rules:
- Answer directly and clearly based on the excerpts.
- For contact details (phone, mobile, email, LinkedIn): look for any number or address format in the excerpts and return it directly.
- A phone/mobile number looks like: +91-XXXXXXXXXX or any digit sequence. If you see one, that IS the answer.
- If the exact information is in the excerpts, state it confidently.
- If the answer is truly not in the excerpts, say: "I'm sorry, I don't see that information in the provided documents."
- Do NOT say information is missing if it IS present in the excerpts.
- Do NOT add commentary like "the name X is not mentioned" if the person is clearly referenced.

Excerpts:
{context}

Question: {query}
Answer:"""


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/", methods=["GET"])
def index():
    uploaded_files = sorted(os.listdir(DATA_DIR)) if os.path.exists(DATA_DIR) else []
    return render_template("index.html", uploaded_files=uploaded_files, ingest_status=ingest_status)


@app.route("/api/search/stream", methods=["POST"])
def api_search_stream():
    data = request.get_json()
    query = (data or {}).get("query", "").strip()
    if not query:
        return jsonify({"error": "Empty query"}), 400

    logger.info(f"[STREAM] Search request: '{query}'")

    chunks, top_k, max_distance = retrieve_chunks(query)

    if not chunks:
        def empty_gen():
            yield f"data: {json.dumps({'chunks': [], 'answer': 'No documents indexed yet or no relevant results found.'})}\n\n"
            yield "data: [DONE]\n\n"
        return Response(stream_with_context(empty_gen()), mimetype="text/event-stream",
                        headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"})

    # Multi-document context window: sort by document order, then expand with adjacent chunks
    chunks = sort_chunks_by_document_order(chunks)
    chunks = fetch_adjacent_chunks(chunks)

    # Attach images from retrieved sources (images rarely match queries semantically,
    # so fetch them by source file when text chunks from that file are retrieved)
    retrieved_sources = {c["source"] for c in chunks}
    existing_image_sources = {c["source"] for c in chunks if c.get("type") == "image"}
    for src in retrieved_sources - existing_image_sources:
        try:
            img_results = collection.get(
                where={"$and": [{"source": src}, {"type": "image"}]},
                include=["documents", "metadatas"],
                limit=5,
            )
            if img_results and img_results["ids"]:
                for i, img_id in enumerate(img_results["ids"]):
                    meta = img_results["metadatas"][i]
                    if meta.get("image_path"):
                        chunks.append({
                            "source": src,
                            "score": 0,
                            "text": img_results["documents"][i],
                            "page": meta.get("page", -1),
                            "type": "image",
                            "image_path": meta.get("image_path"),
                            "chunk_id": img_id,
                        })
        except Exception as e:
            logger.warning(f"[SEARCH] Failed to fetch images for {src}: {e}")

    prompt = build_prompt(chunks, query)

    def generate():
        # Send chunks metadata first
        yield f"data: {json.dumps({'chunks': chunks})}\n\n"
        try:
            logger.info(f"[STREAM] Starting LLM stream for query: '{query}'")
            stream = openai_client.chat.completions.create(
                model=os.getenv("XAI_MODEL", "grok-3-latest"),
                temperature=0,
                stream=True,
                messages=[
                    {"role": "system", "content": "You are a helpful document retrieval assistant."},
                    {"role": "user", "content": prompt}
                ]
            )
            for chunk in stream:
                token = chunk.choices[0].delta.content
                if token:
                    yield f"data: {json.dumps({'token': token})}\n\n"
            logger.info(f"[STREAM] LLM stream completed for query: '{query}'")
        except Exception as e:
            logger.error(f"[STREAM] LLM error for query '{query}': {e}", exc_info=True)
            fallback = "\n\n".join(c["text"].strip() for c in chunks)
            yield f"data: {json.dumps({'token': fallback})}\n\n"
        yield "data: [DONE]\n\n"

    return Response(stream_with_context(generate()), mimetype="text/event-stream",
                    headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"})


@app.route("/api/upload", methods=["POST"])
def upload():
    file = request.files.get("document")
    if file is None or not file.filename:
        return jsonify({"error": "Please select a file first."}), 400

    filename = secure_filename(file.filename)
    if not filename:
        return jsonify({"error": "Invalid filename."}), 400

    if not is_allowed_file(filename):
        return jsonify({"error": "Supported: .txt, .pdf, .docx, .mp4, .mp3, .wav, .m4a, .webm"}), 400

    os.makedirs(DATA_DIR, exist_ok=True)

    if os.path.exists(os.path.join(DATA_DIR, filename)):
        return jsonify({"error": f"'{filename}' already exists. Use Replace to update it."}), 409

    file.save(os.path.join(DATA_DIR, filename))
    logger.info(f"[UPLOAD] File saved: {filename}")

    thread = threading.Thread(target=run_ingest_background, args=(filename,), daemon=True)
    thread.start()

    return jsonify({"filename": filename, "status": "processing"})


@app.route("/ingest_status/<filename>")
def get_ingest_status(filename):
    status = ingest_status.get(filename, "unknown")
    logger.debug(f"[STATUS] {filename} -> {status}")
    return jsonify({"filename": filename, "status": status})


@app.route("/api/delete", methods=["POST"])
def delete():
    data = request.get_json()
    filename = (data or {}).get("filename")
    if filename:
        delete_file_and_index(filename)
        ingest_status.pop(filename, None)
        return jsonify({"ok": True})
    return jsonify({"error": "No file specified."}), 400



@app.errorhandler(413)
def file_too_large(_):
    logger.warning("[UPLOAD] File too large rejected.")
    return jsonify({"error": "File too large. Maximum allowed size is 500 MB."}), 413


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
