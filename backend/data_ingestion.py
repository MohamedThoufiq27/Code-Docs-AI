import logging
import os
from typing import List, Optional, Sequence

from langchain_community.document_loaders import (
    DirectoryLoader,
    TextLoader,
    WebBaseLoader,
)
from langchain_core.documents import Document
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

from .config import Config

logger = logging.getLogger(__name__)

SUPPORTED_GLOBS = ("**/*.md", "**/*.rst", "**/*.txt", "**/*.mdx")

# Stage 1 splits on these markdown headers. The second item in each tuple is
# the metadata key the header text gets stored under, so a chunk under
# "## Authentication" arrives with metadata["h2"] = "Authentication".
HEADERS_TO_SPLIT_ON = [
    ("#", "h1"),
    ("##", "h2"),
    ("###", "h3"),
]


class DocumentIngestion:
    def __init__(self) -> None:
        # Stage 1 — structure-aware split. Each markdown section becomes its
        # own chunk so retrieval doesn't bleed unrelated topics together
        # (e.g. an "Authentication" section won't get glued to "Routing").
        # strip_headers=False keeps the header text inside the chunk so the
        # embedder sees the topic title — a strong retrieval signal.
        self.header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=HEADERS_TO_SPLIT_ON,
            strip_headers=False,
        )
        # Stage 2 — token-bound recursive split. Only sections that exceed
        # CHUNK_SIZE tokens get cut further; smaller sections pass through
        # untouched. Token counting (not chars) keeps chunks predictable
        # against the LLM's context budget.
        self.token_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP,
            separators=["\n\n", "\n", " ", ""],
        )

    def load_local_docs(self, docs_path: str) -> List[Document]:
        if not os.path.isdir(docs_path):
            logger.warning("Skipping missing path: %s", docs_path)
            return []
        documents: List[Document] = []
        for pattern in SUPPORTED_GLOBS:
            loader = DirectoryLoader(
                docs_path,
                glob=pattern,
                loader_cls=TextLoader,
                loader_kwargs={"encoding": "utf-8"},
                silent_errors=True,
                show_progress=False,
            )
            try:
                docs = loader.load()
            except Exception as e:
                logger.warning("Loader failed for %s/%s: %s", docs_path, pattern, e)
                continue
            documents.extend(docs)
        logger.info("Loaded %d docs from %s", len(documents), docs_path)
        return documents

    def load_web_docs(self, urls: Sequence[str]) -> List[Document]:
        documents: List[Document] = []
        for url in urls:
            try:
                docs = WebBaseLoader(url).load()
                documents.extend(docs)
            except Exception as e:
                logger.warning("Failed to load %s: %s", url, e)
        return documents

    def _header_split_one(self, doc: Document) -> List[Document]:
        # Run Stage 1 on a single document. Header-less files (.txt, .rst with
        # === underlines, or .md with no #-style headers) come back as a
        # single section — that's intentional, Stage 2 still tokenizes them.
        try:
            sections = self.header_splitter.split_text(doc.page_content)
        except Exception as e:
            logger.warning(
                "Header split failed for %s: %s", doc.metadata.get("source"), e
            )
            return [doc]
        if not sections:
            return [doc]
        # MarkdownHeaderTextSplitter only emits its own h1/h2/h3 metadata, so
        # re-attach the original file metadata (notably `source`) on top.
        for s in sections:
            s.metadata = {**doc.metadata, **s.metadata}
        return sections

    def chunk_documents(self, documents: Sequence[Document]) -> List[Document]:
        # Stage 1: split every loaded document along its markdown headers.
        sectioned: List[Document] = []
        for doc in documents:
            sectioned.extend(self._header_split_one(doc))

        # Stage 2: token-bound recursive split for any oversized section.
        chunks = self.token_splitter.split_documents(sectioned)

        # Stamp a per-source chunk index so downstream tooling can cite
        # "chunk 3 of intro.md" — handy when debugging retrieval quality.
        per_source_counter: dict = {}
        for c in chunks:
            src = c.metadata.get("source", "unknown")
            idx = per_source_counter.get(src, 0)
            c.metadata["chunk_index"] = idx
            per_source_counter[src] = idx + 1

        logger.info(
            "Chunked %d docs -> %d sections -> %d final chunks",
            len(documents), len(sectioned), len(chunks),
        )
        return chunks

    def prepare_data(
        self,
        docs_paths: Sequence[str],
        web_urls: Optional[Sequence[str]] = None,
    ) -> List[Document]:
        all_docs: List[Document] = []
        for p in docs_paths:
            all_docs.extend(self.load_local_docs(p))
        if web_urls:
            all_docs.extend(self.load_web_docs(web_urls))
        return self.chunk_documents(all_docs)
