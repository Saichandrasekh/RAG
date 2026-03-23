import os
import logging

logger = logging.getLogger(__name__)

BLOB_PREFIX = "chroma_index/"


def _get_container_client():
    from azure.storage.blob import BlobServiceClient
    conn_str = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    container = os.getenv("AZURE_BLOB_CONTAINER", "policyrag-index")
    if not conn_str:
        raise ValueError("AZURE_STORAGE_CONNECTION_STRING not set in .env")
    service = BlobServiceClient.from_connection_string(conn_str)
    # Create container if it doesn't exist
    container_client = service.get_container_client(container)
    try:
        container_client.create_container()
        logger.info(f"[BLOB] Created container: {container}")
    except Exception:
        pass  # already exists
    return container_client


def download_index(local_dir: str) -> bool:
    """Download ChromaDB index from Blob Storage to local_dir on startup."""
    try:
        client = _get_container_client()
        blobs = list(client.list_blobs(name_starts_with=BLOB_PREFIX))

        if not blobs:
            logger.info("[BLOB] No existing index in Blob Storage — starting fresh.")
            return False

        os.makedirs(local_dir, exist_ok=True)

        for blob in blobs:
            relative = blob.name[len(BLOB_PREFIX):]  # strip prefix
            if not relative:
                continue
            local_path = os.path.join(local_dir, relative.replace("/", os.sep))
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            data = client.download_blob(blob.name).readall()
            with open(local_path, "wb") as f:
                f.write(data)

        logger.info(f"[BLOB] Index downloaded: {len(blobs)} files from Blob Storage.")
        return True

    except Exception as e:
        logger.error(f"[BLOB] Download failed: {e}", exc_info=True)
        return False


def upload_index(local_dir: str) -> bool:
    """Upload local ChromaDB index to Blob Storage after ingest."""
    try:
        client = _get_container_client()
        uploaded = 0

        for root, dirs, files in os.walk(local_dir):
            for fname in files:
                local_path = os.path.join(root, fname)
                relative = os.path.relpath(local_path, local_dir).replace("\\", "/")
                blob_name = BLOB_PREFIX + relative
                with open(local_path, "rb") as f:
                    client.upload_blob(blob_name, f, overwrite=True)
                uploaded += 1

        logger.info(f"[BLOB] Index uploaded: {uploaded} files to Blob Storage.")
        return True

    except Exception as e:
        logger.error(f"[BLOB] Upload failed: {e}", exc_info=True)
        return False
