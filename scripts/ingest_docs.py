import logging
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.data_ingestion import DocumentIngestion  # noqa: E402
from backend.vector_store import VectorStore  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Real corpora — clone these via:
#   git clone --depth=1 https://github.com/python/cpython.git    data/cpython
#   git clone --depth=1 https://github.com/tiangolo/fastapi.git  data/fastapi
#   git clone --depth=1 https://github.com/langchain-ai/langchain.git data/langchain
DOC_PATHS = [
    "data/cpython/Doc",
    "data/fastapi/docs",
    "data/langchain/docs",
    "data/docs",  # local sample fallback
]


def main() -> None:
    logger.info("Starting ingestion pipeline")

    os.makedirs("data/docs", exist_ok=True)
    sample = "data/docs/sample_doc.md"
    if not os.path.exists(sample):
        with open(sample, "w", encoding="utf-8") as f:
            f.write("# Sample\nLocal sample doc for smoke testing.\n")

    ingestion = DocumentIngestion()
    chunks = ingestion.prepare_data(DOC_PATHS)
    logger.info("Prepared %d chunks total", len(chunks))

    if not chunks:
        logger.warning("No chunks produced; nothing to ingest.")
        return

    store = VectorStore()
    upserted = store.add_documents(chunks)
    logger.info("Upserted %d chunks into Pinecone", upserted)

    stats = store.index_stats()
    logger.info("Pinecone index stats: %s", stats)
    print("\nFinal index stats:")
    print(stats)


if __name__ == "__main__":
    main()