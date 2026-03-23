from flask import Flask, render_template, request, jsonify, Response, stream_with_context
import chromadb
from chromadb.utils import embedding_functions
import os
import re
import json
import threading
import logging
from dotenv import load_dotenv
from openai import OpenAI
from werkzeug.utils import secure_filename
from ingest import ingest_new_files, DATA_DIR
from blob_sync import download_index, upload_index

# ── Logging setup ────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── App init ─────────────────────────────────────────────────────────────────
load_dotenv()
openai_client = OpenAI(
    api_key=os.getenv("XAI_API_KEY"),
    base_url=os.getenv("XAI_BASE_URL", "https://api.x.ai/v1"),
)

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "dev-secret-key")
ALLOWED_EXTENSIONS = {".txt", ".pdf", ".docx"}

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


# ── Ingest background thread ──────────────────────────────────────────────────
def run_ingest_background(filename):
    logger.info(f"[INGEST] Starting background ingest for: {filename}")
    ingest_status[filename] = "processing"
    try:
        ingest_new_files(collection=collection)
        ingest_status[filename] = "done"
        logger.info(f"[INGEST] Completed successfully: {filename} | Total chunks: {collection.count()}")
        # Upload updated index to Blob Storage so it persists across deployments
        if os.getenv("AZURE_STORAGE_CONNECTION_STRING"):
            logger.info("[BLOB] Uploading updated index to Blob Storage...")
            upload_index(INDEX_DIR)
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
    query_terms = {tok for tok in tokenize(query) if tok not in STOPWORDS}
    ranked = []

    for chunk in raw_chunks:
        dist = chunk["distance"]
        if dist > max_distance:
            continue
        chunk_terms = set(tokenize(chunk["text"]))
        overlap = 0.0
        if query_terms:
            overlap = len(query_terms & chunk_terms) / len(query_terms)
        semantic_score = max(0.0, 1.0 - (dist / max_distance))
        hybrid_score = 0.8 * semantic_score + 0.2 * overlap
        ranked.append({
            "source": chunk["source"],
            "score": round(dist, 3),
            "text": chunk["text"],
            "page": chunk.get("page", -1),
            "hybrid_score": hybrid_score,
        })

    ranked.sort(key=lambda x: (-x["hybrid_score"], x["score"]))

    deduped = []
    seen_texts = set()
    for item in ranked:
        key = " ".join(item["text"].split()).lower()
        if key in seen_texts:
            continue
        seen_texts.add(key)
        item.pop("hybrid_score", None)
        deduped.append(item)
        if len(deduped) >= top_k:
            break

    if deduped:
        return deduped

    raw_by_distance = sorted(raw_chunks, key=lambda x: x["distance"])
    return [
        {"source": i["source"], "score": round(i["distance"], 3),
         "text": i["text"], "page": i.get("page", -1)}
        for i in raw_by_distance[:top_k]
    ]


def retrieve_chunks(query, top_k=None):
    if top_k is None:
        top_k = int(os.getenv("RETRIEVAL_TOP_K", "8"))
    max_distance = float(os.getenv("RETRIEVAL_MAX_DISTANCE", "1.4"))

    total_docs = collection.count()
    logger.info(f"[SEARCH] Query: '{query}' | Total indexed chunks: {total_docs}")

    if total_docs == 0:
        logger.warning("[SEARCH] No documents in index.")
        return [], top_k, max_distance

    initial_k = min(max(top_k * 4, 20), total_docs)
    results = collection.query(query_texts=[query], n_results=initial_k)

    raw_chunks = []
    if results and results['documents'] and results['documents'][0]:
        for i in range(len(results['documents'][0])):
            meta = results['metadatas'][0][i]
            dist = float(results['distances'][0][i])
            raw_chunks.append({
                "source": meta.get("source", "Unknown"),
                "distance": dist,
                "text": results['documents'][0][i],
                "page": meta.get("page", -1),
            })

    chunks = rerank_and_filter_chunks(raw_chunks, query, top_k=top_k, max_distance=max_distance)
    logger.info(f"[SEARCH] Returning {len(chunks)} chunks for query: '{query}'")
    return chunks, top_k, max_distance


def build_prompt(chunks, query):
    context = "\n\n".join([c["text"].strip() for c in chunks])
    return f"""You are a helpful, intelligent Document Retrieval Assistant.
Please read the provided Excerpts and answer the User's Question clearly and conversationally.
- Use ONLY the provided Excerpts.
- If the answer is not contained in the Excerpts, simply reply: "I'm sorry, I don't see the answer to that in the provided documents."
- If the Excerpts are not relevant to the question, do not guess.

Excerpts:
{context}

User's Question: {query}
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
        return jsonify({"error": "Only .txt, .pdf, and .docx files are supported."}), 400

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
