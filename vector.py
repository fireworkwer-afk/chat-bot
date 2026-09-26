"""
vector.py
---------
Handles ingestion of local knowledge-base data (reviews.csv or .txt files)
into a persistent ChromaDB vector store using Ollama embeddings
(model: mxbai-embed-large).

Usage:
    python vector.py                # build only if DB doesn't exist yet
    python vector.py --rebuild      # force a full rebuild of the DB

This module also exposes `get_retriever()` so that main.py can import
and reuse the same persisted Chroma collection for offline similarity
search, without re-embedding data on every app run.
"""

import os
import sys
import argparse
import logging
import glob

import pandas as pd
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
EMBEDDING_MODEL = "mxbai-embed-large"
DB_LOCATION = "./chroma_langchain_db"
COLLECTION_NAME = "local_knowledge_base"
CSV_PATH = "reviews.csv"
TXT_DATA_DIR = "./data"

# --------------------------------------------------------------------------
# Logging setup - step-by-step terminal output
# --------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("vector")


def load_documents_from_csv(csv_path: str) -> list[Document]:
    """
    Load a reviews-style CSV into LangChain Document objects.
    Expects flexible columns; falls back gracefully if some are missing.
    """
    logger.info(f"Reading CSV data source: {csv_path}")
    df = pd.read_csv(csv_path)

    documents = []
    for i, row in df.iterrows():
        # Build page content out of whatever text-like columns exist.
        title = str(row.get("Title", "")).strip()
        body = str(row.get("Review", row.get("Content", row.get("Text", "")))).strip()
        rating = row.get("Rating", "")
        date = row.get("Date", "")

        page_content = f"{title}\n{body}".strip()
        if not page_content:
            continue

        metadata = {
            "source": csv_path,
            "row": int(i),
            "rating": str(rating),
            "date": str(date),
        }
        documents.append(Document(page_content=page_content, metadata=metadata))

    logger.info(f"Loaded {len(documents)} documents from CSV.")
    return documents


def load_documents_from_txt(txt_dir: str) -> list[Document]:
    """
    Load all .txt files inside a directory into LangChain Document objects.
    Each file becomes one document (simple, robust default).
    """
    logger.info(f"Scanning for .txt files in: {txt_dir}")
    txt_files = glob.glob(os.path.join(txt_dir, "*.txt"))

    documents = []
    for path in txt_files:
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read().strip()
            if content:
                documents.append(
                    Document(page_content=content, metadata={"source": path})
                )
        except Exception as e:
            logger.warning(f"Could not read {path}: {e}")

    logger.info(f"Loaded {len(documents)} documents from TXT files.")
    return documents


def load_all_documents() -> list[Document]:
    """
    Decide which data source to use:
      1. reviews.csv if present
      2. otherwise any .txt files under ./data/
    Returns a combined list of Documents. Raises if nothing is found.
    """
    documents: list[Document] = []

    if os.path.exists(CSV_PATH):
        documents.extend(load_documents_from_csv(CSV_PATH))
    else:
        logger.info(f"No {CSV_PATH} found, skipping CSV ingestion.")

    if os.path.isdir(TXT_DATA_DIR):
        documents.extend(load_documents_from_txt(TXT_DATA_DIR))
    else:
        logger.info(f"No {TXT_DATA_DIR} directory found, skipping TXT ingestion.")

    if not documents:
        logger.error(
            "No data sources found. Place a 'reviews.csv' file in the project "
            "root, or .txt files inside a './data' directory, then re-run."
        )
        raise FileNotFoundError("No ingestible data found for vector.py")

    return documents


def build_vector_store(rebuild: bool = False) -> Chroma:
    """
    Build (or load) the persistent Chroma vector store.

    If `rebuild` is True, or the DB does not exist yet, documents are
    freshly loaded, embedded, and added to a new/rebuilt collection.
    Otherwise, the existing persisted collection is simply loaded.
    """
    db_exists = os.path.exists(DB_LOCATION) and os.listdir(DB_LOCATION)

    logger.info("Initializing Ollama embedding model: %s", EMBEDDING_MODEL)
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

    need_to_embed = rebuild or not db_exists

    if need_to_embed:
        logger.info("Building vector store from scratch (rebuild=%s, existing=%s)",
                     rebuild, db_exists)
        documents = load_all_documents()
        ids = [str(i) for i in range(len(documents))]

        logger.info("Creating Chroma collection and embedding %d documents...",
                    len(documents))
        vector_store = Chroma(
            collection_name=COLLECTION_NAME,
            persist_directory=DB_LOCATION,
            embedding_function=embeddings,
        )

        # If rebuilding an existing collection, clear it first.
        if db_exists and rebuild:
            try:
                logger.info("Clearing existing collection before rebuild...")
                existing_ids = vector_store.get()["ids"]
                if existing_ids:
                    vector_store.delete(ids=existing_ids)
            except Exception as e:
                logger.warning(f"Could not clear old collection cleanly: {e}")

        vector_store.add_documents(documents=documents, ids=ids)
        logger.info("Embedding complete. Vector store persisted at: %s", DB_LOCATION)
    else:
        logger.info("Existing Chroma DB found at %s. Loading without re-embedding.",
                    DB_LOCATION)
        vector_store = Chroma(
            collection_name=COLLECTION_NAME,
            persist_directory=DB_LOCATION,
            embedding_function=embeddings,
        )

    return vector_store


def get_retriever(k: int = 3):
    """
    Convenience accessor used by main.py to get a similarity-search
    retriever over the persisted Chroma collection, without triggering
    a rebuild. Wrapped by the caller in try/except for robustness.
    """
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=DB_LOCATION,
        embedding_function=embeddings,
    )
    return vector_store.as_retriever(search_kwargs={"k": k})


def main():
    parser = argparse.ArgumentParser(description="Build/refresh the local Chroma vector store.")
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Force a full rebuild of the vector store from source data.",
    )
    args = parser.parse_args()

    logger.info("=== vector.py started ===")
    try:
        build_vector_store(rebuild=args.rebuild)
        logger.info("=== Vector store ready. Done. ===")
    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)
    except Exception as e:
        logger.exception(f"Unexpected error while building vector store: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()