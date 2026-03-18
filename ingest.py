import os
import sys
import logging
from tqdm import tqdm
import chromadb
from chromadb.utils import embedding_functions
from utils import yield_file_chunks

logger = logging.getLogger(__name__)

DATA_DIR = "data/raw/knowledge"
INDEX_DIR = "index/chroma_db"
IMAGES_DIR = "static/images"
COLLECTION_NAME = "knowledge_base"
BATCH_SIZE = 100


def log(msg):
    """Print to stdout immediately (visible in Azure Log Stream)."""
    print(msg, flush=True)
    logger.info(msg)


def ingest_new_files(data_dir=DATA_DIR, index_dir=INDEX_DIR, collection=None):
    if collection is None:
        os.makedirs(index_dir, exist_ok=True)
        log("[INGEST] No collection passed — initializing ChromaDB...")
        chroma_client = chromadb.PersistentClient(path=index_dir)
        sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        collection = chroma_client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=sentence_transformer_ef,
        )

    log("[INGEST] Fetching already indexed sources from ChromaDB...")
    existing_sources = set()
    existing_docs = collection.get(include=["metadatas"])
    if existing_docs and existing_docs["metadatas"]:
        for meta in existing_docs["metadatas"]:
            if meta and "source" in meta:
                existing_sources.add(meta["source"])

    log(f"[INGEST] Already indexed: {len(existing_sources)} sources | Total chunks in DB: {collection.count()}")

    all_files = os.listdir(data_dir)
    pending = [f for f in all_files if f not in existing_sources]
    log(f"[INGEST] Total files in data dir: {len(all_files)} | Pending (not yet indexed): {len(pending)}")

    if not pending:
        log("[INGEST] Nothing to do. All files already indexed.")
        return False

    new_files_added = False

    for file_num, filename in enumerate(pending, start=1):
        path = os.path.join(data_dir, filename)
        file_size_mb = round(os.path.getsize(path) / (1024 * 1024), 2)
        log(f"[INGEST] ({file_num}/{len(pending)}) Starting: {filename} | Size: {file_size_mb} MB")

        try:
            log(f"[INGEST] Chunking file: {filename}")
            chunk_generator = yield_file_chunks(path)
            if chunk_generator is None:
                log(f"[INGEST] WARNING: Unsupported file type, skipping: {filename}")
                continue

            batch_documents = []
            batch_metadatas = []
            batch_ids = []
            chunk_idx = 0
            batch_num = 0

            for chunk, page_num in chunk_generator:
                if not chunk.strip():
                    continue
                batch_documents.append(chunk)
                batch_metadatas.append({
                    "source": filename,
                    "page": page_num if page_num is not None else -1
                })
                batch_ids.append(f"{filename}_chunk_{chunk_idx}")
                chunk_idx += 1

                if len(batch_documents) >= BATCH_SIZE:
                    batch_num += 1
                    log(f"[INGEST] Upserting batch #{batch_num} ({len(batch_documents)} chunks) for: {filename}")
                    collection.upsert(
                        documents=batch_documents,
                        metadatas=batch_metadatas,
                        ids=batch_ids,
                    )
                    log(f"[INGEST] Batch #{batch_num} done for: {filename}")
                    batch_documents = []
                    batch_metadatas = []
                    batch_ids = []
                    new_files_added = True

            # Final partial batch
            if batch_documents:
                batch_num += 1
                log(f"[INGEST] Upserting final batch #{batch_num} ({len(batch_documents)} chunks) for: {filename}")
                collection.upsert(
                    documents=batch_documents,
                    metadatas=batch_metadatas,
                    ids=batch_ids,
                )
                log(f"[INGEST] Final batch done for: {filename}")
                new_files_added = True

            log(f"[INGEST] COMPLETED: {filename} | Total chunks: {chunk_idx} | Batches: {batch_num} | DB total: {collection.count()}")

        except MemoryError:
            log(f"[INGEST] MEMORY ERROR on {filename} — file too large for available RAM. Skipping.")
            logger.error(f"[INGEST] MemoryError on {filename}", exc_info=True)
            continue
        except Exception as e:
            log(f"[INGEST] ERROR on {filename}: {type(e).__name__}: {e}")
            logger.error(f"[INGEST] Exception on {filename}", exc_info=True)
            continue

    if new_files_added:
        log(f"[INGEST] All done. Final DB total chunks: {collection.count()}")
    else:
        log("[INGEST] No new files were added.")

    return new_files_added


if __name__ == "__main__":
    ingest_new_files()
