import logging
import os
import random
import statistics
import sys
from typing import List

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.monitoring import query_logger  # noqa: E402
from backend.rag_pipeline import RAGPipeline  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

UNIQUE_QUERIES = [
    "How do I create a REST API in FastAPI?",
    "What is dependency injection in FastAPI?",
    "How does async/await work in Python?",
    "How do I handle exceptions in FastAPI?",
    "How do I add CORS middleware in FastAPI?",
    "What is a Pydantic BaseModel?",
    "How do I write a list comprehension in Python?",
    "How do I structure a LangChain RAG chain?",
    "What are LangChain runnables?",
    "How do I use Pinecone for vector search?",
    "What is a context manager in Python?",
    "How do I read a file safely in Python?",
    "How do I run pytest with markers?",
    "What is dataclass in Python?",
    "How do I cache function results in Python?",
]

REPEAT_FRACTION = 0.30


def build_test_set(seed: int = 42) -> List[str]:
    rng = random.Random(seed)
    n_repeats = int(len(UNIQUE_QUERIES) * REPEAT_FRACTION)
    repeats = rng.choices(UNIQUE_QUERIES, k=n_repeats)
    full = UNIQUE_QUERIES + repeats
    rng.shuffle(full)
    return full


def percentiles(values: List[float]):
    s = sorted(values)

    def pct(p: float) -> float:
        if not s:
            return 0.0
        k = (len(s) - 1) * (p / 100)
        lo, hi = int(k), min(int(k) + 1, len(s) - 1)
        if lo == hi:
            return s[lo]
        return s[lo] + (s[hi] - s[lo]) * (k - lo)

    return pct(50), pct(95), pct(99)


def run_pass(rag: RAGPipeline, queries: List[str], use_cache: bool) -> dict:
    query_logger.reset()
    rag.clear_cache()
    latencies: List[float] = []
    for i, q in enumerate(queries, 1):
        result = rag.query(q, use_cache=use_cache)
        latencies.append(result["latency_ms"])
        logger.info("[%d/%d] cached=%s latency=%.1fms",
                    i, len(queries), result["cached"], result["latency_ms"])
    p50, p95, p99 = percentiles(latencies)
    return {
        "total_queries": len(queries),
        "total_cost_usd": query_logger.total_cost,
        "p50_ms": p50,
        "p95_ms": p95,
        "p99_ms": p99,
        "latencies_ms": latencies,
    }


def main() -> None:
    queries = build_test_set()
    logger.info("Built test set with %d queries (%d unique, %d repeats)",
                len(queries), len(UNIQUE_QUERIES), len(queries) - len(UNIQUE_QUERIES))

    rag = RAGPipeline()

    logger.info("\n=== Pass 1: cache OFF ===")
    no_cache = run_pass(rag, queries, use_cache=False)

    logger.info("\n=== Pass 2: cache ON ===")
    with_cache = run_pass(rag, queries, use_cache=True)

    if no_cache["total_cost_usd"] > 0:
        cost_reduction_pct = (
            (no_cache["total_cost_usd"] - with_cache["total_cost_usd"])
            / no_cache["total_cost_usd"]
            * 100
        )
    else:
        cost_reduction_pct = 0.0

    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)
    print(f"Test set:                {len(queries)} queries "
          f"({len(UNIQUE_QUERIES)} unique, "
          f"{len(queries) - len(UNIQUE_QUERIES)} repeats)")
    print()
    print(f"Cost no-cache:           ${no_cache['total_cost_usd']:.6f}")
    print(f"Cost with-cache:         ${with_cache['total_cost_usd']:.6f}")
    print(f"Cost reduction:          {cost_reduction_pct:.1f}%")
    print()
    print("Latency (no-cache pass — represents real user experience):")
    print(f"  p50:                   {no_cache['p50_ms']:.1f} ms")
    print(f"  p95:                   {no_cache['p95_ms']:.1f} ms")
    print(f"  p99:                   {no_cache['p99_ms']:.1f} ms")
    print()
    print("Latency (with-cache pass):")
    print(f"  p50:                   {with_cache['p50_ms']:.1f} ms")
    print(f"  p95:                   {with_cache['p95_ms']:.1f} ms")
    print(f"  p99:                   {with_cache['p99_ms']:.1f} ms")
    print("=" * 60)


if __name__ == "__main__":
    main()