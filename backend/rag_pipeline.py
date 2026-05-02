import hashlib
import json
import logging
import re
import time
from typing import List, Tuple

from cachetools import TTLCache
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from sentence_transformers import CrossEncoder

from .config import Config
from .monitoring import count_tokens, query_logger
from .vector_store import VectorStore

logger = logging.getLogger(__name__)

PROMPT_TEMPLATE = """You are an expert code documentation assistant.

Use the documentation context below to answer the user's question.
Respond with valid JSON containing exactly these fields:
  - "answer": a direct, concise explanation
  - "code_example": a runnable code snippet, or "" if not applicable
  - "key_points": a list of strings with 2-5 short bullets

Documentation Context:
{context}

User Question: {question}

JSON Answer:"""


def _normalize_key(question: str) -> str:
    q = question.strip().lower()
    q = re.sub(r"[?.!\s]+$", "", q)
    q = re.sub(r"\s+", " ", q)
    return hashlib.md5(q.encode("utf-8")).hexdigest()


class RAGPipeline:
    def __init__(self) -> None:
        self.vector_store = VectorStore()
        self.llm = ChatOpenAI(
            model=Config.LLM_MODEL,
            api_key=Config.OPENROUTER_API_KEY,
            base_url="https://openrouter.ai/api/v1",
            temperature=0.3,
            max_tokens=1000,
        )
        self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        self.prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        self.parser = StrOutputParser()
        self.query_cache: TTLCache = TTLCache(
            maxsize=Config.CACHE_MAX_SIZE,
            ttl=Config.CACHE_TTL,
        )

    def _retrieve(self, question: str) -> List[Document]:
        candidates = self.vector_store.similarity_search(question, k=Config.INITIAL_K)
        if not candidates:
            return []
        pairs = [(question, doc.page_content) for doc in candidates]
        scores = self.reranker.predict(pairs)
        ranked: List[Tuple[float, Document]] = sorted(
            zip(scores, candidates), key=lambda x: float(x[0]), reverse=True
        )
        return [doc for _score, doc in ranked[: Config.FINAL_K]]

    def _format_context(self, docs: List[Document]) -> str:
        blocks = [
            f"Source: {d.metadata.get('source', 'Unknown')}\n{d.page_content}"
            for d in docs
        ]
        ctx = "\n\n---\n\n".join(blocks)
        return ctx[: Config.MAX_CONTEXT_LENGTH]

    def _parse_response(self, raw: str) -> dict:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        return {"answer": raw, "code_example": "", "key_points": []}

    def query(self, question: str, use_cache: bool = True) -> dict:
        start = time.time()
        cache_key = _normalize_key(question)

        if use_cache and Config.CACHE_ENABLED and cache_key in self.query_cache:
            cached = self.query_cache[cache_key]
            latency_ms = (time.time() - start) * 1000
            query_logger.log(
                question=question,
                latency_ms=latency_ms,
                docs_retrieved=len(cached["sources"]),
                input_tokens=0,
                output_tokens=0,
                cached=True,
            )
            return {
                "response": cached["response"],
                "sources": cached["sources"],
                "latency_ms": latency_ms,
                "cached": True,
            }

        docs = self._retrieve(question)
        context = self._format_context(docs)

        chain = self.prompt | self.llm | self.parser
        prompt_text = self.prompt.format(context=context, question=question)
        raw = chain.invoke({"context": context, "question": question})

        parsed = self._parse_response(raw)
        sources = [d.metadata.get("source", "Unknown") for d in docs]
        latency_ms = (time.time() - start) * 1000

        input_tokens = count_tokens(prompt_text)
        output_tokens = count_tokens(raw)

        query_logger.log(
            question=question,
            latency_ms=latency_ms,
            docs_retrieved=len(docs),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached=False,
        )

        result = {
            "response": parsed,
            "sources": sources,
            "latency_ms": latency_ms,
            "cached": False,
        }
        if Config.CACHE_ENABLED:
            self.query_cache[cache_key] = {"response": parsed, "sources": sources}
        return result

    def clear_cache(self) -> None:
        self.query_cache.clear()