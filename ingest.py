import os
import logging
from tqdm import tqdm
import chromadb
from chromadb.utils import embedding_functions
from utils import yield_file_chunks, extract_pdf_images

logger = logging.getLogger(__name__)

DATA_DIR = "data/raw/knowledge"
INDEX_DIR = "index/chroma_db"
IMAGES_DIR = "static/images"
COLLECTION_NAME = "knowledge_base"
BATCH_SIZE = 100


def ingest_new_files(data_dir=DATA_DIR, index_dir=INDEX_DIR, collection=None):
    if collection is None:
        os.makedirs(index_dir, exist_ok=True)
        print("Initializing ChromaDB...")
        chroma_client = chromadb.PersistentClient(path=index_dir)
        sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        collection = chroma_client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=sentence_transformer_ef,
        )

    existing_sources = set()
    existing_docs = collection.get(include=["metadatas"])
    if existing_docs and existing_docs["metadatas"]:
        for meta in existing_docs["metadatas"]:
            if meta and "source" in meta:
                existing_sources.add(meta["source"])

    logger.info(f"[INGEST] Already indexed sources: {len(existing_sources)}")

    new_files_added = False
    all_files = os.listdir(data_dir)
    pending = [f for f in all_files if f not in existing_sources]
    logger.info(f"[INGEST] Files to process: {len(pending)} / {len(all_files)}")

    for filename in tqdm(pending):
        path = os.path.join(data_dir, filename)
        logger.info(f"[INGEST] Processing: {filename}")

        try:
            # Extract page images for PDFs
            if filename.lower().endswith(".pdf"):
                img_dir = os.path.join(IMAGES_DIR, filename)
                logger.info(f"[INGEST] Extracting images from PDF: {filename}")
                extract_pdf_images(path, img_dir)

            chunk_generator = yield_file_chunks(path)
            if chunk_generator is None:
                logger.warning(f"[INGEST] Unsupported file type, skipping: {filename}")
                continue

            batch_documents = []
            batch_metadatas = []
            batch_ids = []
            chunk_idx = 0

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
                    collection.upsert(
                        documents=batch_documents,
                        metadatas=batch_metadatas,
                        ids=batch_ids,
                    )
                    logger.info(f"[INGEST] Upserted batch of {len(batch_documents)} chunks for: {filename}")
                    batch_documents = []
                    batch_metadatas = []
                    batch_ids = []
                    new_files_added = True

            if batch_documents:
                collection.upsert(
                    documents=batch_documents,
                    metadatas=batch_metadatas,
                    ids=batch_ids,
                )
                logger.info(f"[INGEST] Upserted final batch of {len(batch_documents)} chunks for: {filename}")
                new_files_added = True

            logger.info(f"[INGEST] Done: {filename} | Total chunks: {chunk_idx}")

        except Exception as e:
            logger.error(f"[INGEST] ERROR processing {filename}: {e}", exc_info=True)
            continue  # skip this file, continue with others

    if not new_files_added:
        logger.info("[INGEST] No new files to ingest. Index is up to date.")
    else:
        logger.info(f"[INGEST] Index updated. Total chunks in DB: {collection.count()}")

    return new_files_added


if __name__ == "__main__":
    ingest_new_files()
