from flask import Flask, render_template, request, redirect, url_for
import chromadb
from chromadb.utils import embedding_functions
import os
import re
import shutil
from dotenv import load_dotenv
from openai import OpenAI
from werkzeug.utils import secure_filename
from ingest import ingest_new_files, DATA_DIR, IMAGES_DIR

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

# Initialize ChromaDB
print("Initializing ChromaDB in app...")
chroma_client = chromadb.PersistentClient(path="index/chroma_db")
sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
collection = chroma_client.get_or_create_collection(
    name="knowledge_base",
    embedding_function=sentence_transformer_ef
)


def is_allowed_file(filename):
    return os.path.splitext(filename.lower())[1] in ALLOWED_EXTENSIONS


def delete_file_and_index(filename):
    file_path = os.path.join(DATA_DIR, filename)
    if os.path.exists(file_path):
        os.remove(file_path)
    results = collection.get(where={"source": filename})
    if results and results["ids"]:
        collection.delete(ids=results["ids"])
    img_dir = os.path.join(IMAGES_DIR, filename)
    if os.path.exists(img_dir):
        shutil.rmtree(img_dir)

def generate_rag_answer(chunks, query):
    if not chunks:
        return "Sorry, I couldn't find any relevant documents to answer your question."

    # Combine all retrieved text to serve as context
    context = "\n\n".join([chunk["text"].strip() for chunk in chunks])

    prompt = f"""You are a helpful, intelligent Document Retrieval Assistant. 
Please read the provided Excerpts and answer the User's Question clearly and conversationally.
- Use ONLY the provided Excerpts.
- If the answer is not contained in the Excerpts, simply reply: "I'm sorry, I don't see the answer to that in the provided documents."
- If the Excerpts are not relevant to the question, do not guess.

Excerpts:
{context}

User's Question: {query}
Answer:"""

    try:
        response = openai_client.chat.completions.create(
            model=os.getenv("XAI_MODEL", "grok-3-latest"),
            temperature=0,
            messages=[
                {"role": "system", "content": "You are a helpful document retrieval assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content
        
    except Exception as e:
        print(f"LLM Generation Error: {e}")
        # Fallback to the raw chunks if the API call fails
        seen = set()
        paragraphs = []
        for chunk in chunks:
            text = chunk["text"].strip()
            if text and text not in seen:
                seen.add(text)
                paragraphs.append(text)
        return "\n\n".join(paragraphs)

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

        # Reject only very weak matches; keep moderately related context.
        if overlap == 0 and dist > (max_distance * 0.9):
            continue

        ranked.append(
            {
                "source": chunk["source"],
                "score": round(dist, 3),
                "text": chunk["text"],
                "page": chunk.get("page", -1),
                "image_urls": chunk.get("image_urls", []),
                "hybrid_score": hybrid_score,
            }
        )

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

    # Fallback: if filters were too strict, return nearest chunks by distance.
    raw_by_distance = sorted(raw_chunks, key=lambda x: x["distance"])
    fallback = []
    for item in raw_by_distance[:top_k]:
        fallback.append(
            {
                "source": item["source"],
                "score": round(item["distance"], 3),
                "text": item["text"],
                "page": item.get("page", -1),
                "image_urls": item.get("image_urls", []),
            }
        )
    return fallback

def search(query, top_k=None):
    if top_k is None:
        top_k = int(os.getenv("RETRIEVAL_TOP_K", "8"))

    initial_k = max(top_k * 4, 20)
    max_distance = float(os.getenv("RETRIEVAL_MAX_DISTANCE", "1.4"))

    results = collection.query(
        query_texts=[query],
        n_results=initial_k
    )

    raw_chunks = []
    if results and results['documents'] and results['documents'][0]:
        for i in range(len(results['documents'][0])):
            doc_text = results['documents'][0][i]
            meta = results['metadatas'][0][i]
            dist = float(results['distances'][0][i])
            source = meta.get("source", "Unknown")
            page = meta.get("page", -1)

            image_url = None
            if page >= 0:
                img_dir = os.path.join("static", "images", source)
                if os.path.exists(img_dir):
                    matches = [
                        f"/static/images/{source}/{f}"
                        for f in sorted(os.listdir(img_dir))
                        if f.startswith(f"page_{page}_")
                    ]
                    image_url = matches if matches else None

            raw_chunks.append({
                "source": source,
                "distance": dist,
                "text": doc_text,
                "page": page,
                "image_urls": image_url or [],
            })

    chunks = rerank_and_filter_chunks(raw_chunks, query, top_k=top_k, max_distance=max_distance)
    answer = generate_rag_answer(chunks, query)

    return {
        "answer": answer,
        "chunks": chunks
    }

@app.route("/", methods=["GET", "POST"])
def index():
    query = ""
    answer = ""
    chunks = []
    upload_status = request.args.get("upload_status", "")

    if request.method == "POST":
        query = request.form.get("query", "").strip()
        if query:
            result = search(query)
            answer = result.get("answer", "")
            chunks = result.get("chunks", [])

    uploaded_files = sorted(os.listdir(DATA_DIR)) if os.path.exists(DATA_DIR) else []

    return render_template(
        "index.html",
        query=query,
        answer=answer,
        chunks=chunks,
        upload_status=upload_status,
        uploaded_files=uploaded_files
    )


@app.route("/upload", methods=["POST"])
def upload():
    file = request.files.get("document")
    if file is None or not file.filename:
        return redirect(url_for("index", upload_status="Please select a file first."))

    filename = secure_filename(file.filename)
    if not filename:
        return redirect(url_for("index", upload_status="Invalid filename."))

    if not is_allowed_file(filename):
        return redirect(url_for("index", upload_status="Only .txt, .pdf, and .docx files are supported."))

    os.makedirs(DATA_DIR, exist_ok=True)

    if os.path.exists(os.path.join(DATA_DIR, filename)):
        return redirect(url_for("index", upload_status=f"'{filename}' already exists. Use Replace to update it."))

    file.save(os.path.join(DATA_DIR, filename))

    try:
        ingest_new_files()
        message = f"Uploaded '{filename}' and updated the index."
    except Exception as e:
        message = f"File uploaded, but ingest failed: {e}"

    return redirect(url_for("index", upload_status=message))


@app.route("/delete", methods=["POST"])
def delete():
    filename = request.form.get("filename")
    if filename:
        delete_file_and_index(filename)
        message = f"Deleted '{filename}'."
    else:
        message = "No file specified."
    return redirect(url_for("index", upload_status=message))


@app.route("/replace", methods=["POST"])
def replace():
    file = request.files.get("document")
    filename = request.form.get("filename")

    if not file or not filename:
        return redirect(url_for("index", upload_status="Missing file or filename."))

    delete_file_and_index(filename)
    os.makedirs(DATA_DIR, exist_ok=True)
    file.save(os.path.join(DATA_DIR, filename))

    try:
        ingest_new_files()
        message = f"Replaced '{filename}' and updated the index."
    except Exception as e:
        message = f"File replaced, but ingest failed: {e}"

    return redirect(url_for("index", upload_status=message))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
