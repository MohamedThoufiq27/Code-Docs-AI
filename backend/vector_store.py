import hashlib
import logging
from typing import Iterable, List, Optional

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

from .config import Config

logger = logging.getLogger(__name__)


def _chunk_id(doc: Document) -> str:
    source = str(doc.metadata.get("source", ""))
    payload = f"{source}\x00{doc.page_content}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class VectorStore:
    def __init__(self) -> None:
        self.index_name = Config.PINECONE_INDEX_NAME
        self.pc = Pinecone(api_key=Config.PINECONE_API_KEY)
        self.embeddings = OpenAIEmbeddings(
            api_key=Config.OPENAI_API_KEY,
            model=Config.EMBEDDING_MODEL,
        )
        self._ensure_index_exists()
        self._vectorstore: Optional[PineconeVectorStore] = None

    def _ensure_index_exists(self) -> None:
        existing = {idx["name"] for idx in self.pc.list_indexes()}
        if self.index_name not in existing:
            logger.info("Creating Pinecone index %s", self.index_name)
            self.pc.create_index(
                name=self.index_name,
                dimension=Config.PINECONE_DIMENSION,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud=Config.PINECONE_CLOUD,
                    region=Config.PINECONE_REGION,
                ),
            )

    def get_vectorstore(self) -> PineconeVectorStore:
        if self._vectorstore is None:
            self._vectorstore = PineconeVectorStore(
                index_name=self.index_name,
                embedding=self.embeddings,
            )
        return self._vectorstore

    def add_documents(self, chunks: List[Document], batch_size: int = 100) -> int:
        if not chunks:
            return 0
        vs = self.get_vectorstore()
        ids = [_chunk_id(c) for c in chunks]
        total = 0
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            batch_ids = ids[start : start + batch_size]
            vs.add_documents(documents=batch, ids=batch_ids)
            total += len(batch)
            logger.info("Upserted batch: %d/%d", total, len(chunks))
        return total

    def similarity_search(self, query: str, k: int = 5) -> List[Document]:
        return self.get_vectorstore().similarity_search(query, k=k)

    def as_retriever(self, k: int = 5):
        return self.get_vectorstore().as_retriever(search_kwargs={"k": k})

    def index_stats(self) -> dict:
        stats = self.pc.Index(self.index_name).describe_index_stats()
        if hasattr(stats, "to_dict"):
            return stats.to_dict()
        return dict(stats)