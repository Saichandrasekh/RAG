import os
from tqdm import tqdm
import chromadb
from chromadb.utils import embedding_functions
from utils import yield_file_chunks

DATA_DIR = "data/raw/knowledge"
INDEX_DIR = "index/chroma_db"
COLLECTION_NAME = "knowledge_base"
BATCH_SIZE = 100


def ingest_new_files(data_dir=DATA_DIR, index_dir=INDEX_DIR):
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

    existing_docs = collection.get(include=["metadatas"])
    existing_sources = set()
    if existing_docs and existing_docs["metadatas"]:
        for meta in existing_docs["metadatas"]:
            if meta and "source" in meta:
                existing_sources.add(meta["source"])

    new_files_added = False

    for filename in tqdm(os.listdir(data_dir)):
        if filename in existing_sources:
            continue

        path = os.path.join(data_dir, filename)
        chunk_generator = yield_file_chunks(path)
        if chunk_generator is None:
            continue

        batch_documents = []
        batch_metadatas = []
        batch_ids = []

        chunk_idx = 0
        for chunk in chunk_generator:
            if not chunk.strip():
                continue

            batch_documents.append(chunk)
            batch_metadatas.append({"source": filename})
            batch_ids.append(f"{filename}_chunk_{chunk_idx}")
            chunk_idx += 1

            if len(batch_documents) >= BATCH_SIZE:
                collection.upsert(
                    documents=batch_documents,
                    metadatas=batch_metadatas,
                    ids=batch_ids,
                )
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
            new_files_added = True

    if not new_files_added:
        print("No new files to ingest. Index is up to date.")
    else:
        print("Index updated successfully.")
        print("Total chunks in database:", collection.count())

    return new_files_added


if __name__ == "__main__":
    ingest_new_files()
