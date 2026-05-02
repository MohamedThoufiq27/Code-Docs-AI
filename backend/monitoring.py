import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import tiktoken

from .config import Config

LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)

# tiktoken doesn't ship encodings for every router-namespaced model name
# (e.g. "openai/gpt-3.5-turbo"). cl100k_base is the encoding GPT-3.5/4 use.
try:
    _ENC = tiktoken.encoding_for_model("gpt-3.5-turbo")
except KeyError:
    _ENC = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    if not text:
        return 0
    return len(_ENC.encode(text))


def estimate_cost(input_tokens: int, output_tokens: int) -> float:
    in_cost = (input_tokens / 1_000_000) * Config.COST_PER_1M_INPUT
    out_cost = (output_tokens / 1_000_000) * Config.COST_PER_1M_OUTPUT
    return in_cost + out_cost


@dataclass
class QueryRecord:
    timestamp: str
    question: str
    latency_ms: float
    docs_retrieved: int
    input_tokens: int
    output_tokens: int
    cost_usd: float
    cached: bool


class QueryLogger:
    def __init__(self) -> None:
        self.records: List[QueryRecord] = []

    @property
    def total_cost(self) -> float:
        return sum(r.cost_usd for r in self.records)

    @property
    def total_queries(self) -> int:
        return len(self.records)

    def log(
        self,
        question: str,
        latency_ms: float,
        docs_retrieved: int,
        input_tokens: int,
        output_tokens: int,
        cached: bool,
    ) -> QueryRecord:
        cost = 0.0 if cached else estimate_cost(input_tokens, output_tokens)
        rec = QueryRecord(
            timestamp=datetime.now(timezone.utc).isoformat(),
            question=question[:200],
            latency_ms=latency_ms,
            docs_retrieved=docs_retrieved,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
            cached=cached,
        )
        self.records.append(rec)
        logger.info(
            "query latency=%.1fms docs=%d in=%d out=%d cost=$%.6f cached=%s",
            latency_ms, docs_retrieved, input_tokens, output_tokens, cost, cached,
        )
        return rec

    def reset(self) -> None:
        self.records = []

    def save(self, path: Optional[Path] = None) -> Path:
        path = path or LOGS_DIR / f"queries_{datetime.now().strftime('%Y%m%d')}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "records": [asdict(r) for r in self.records],
                    "total_cost": self.total_cost,
                    "total_queries": self.total_queries,
                },
                f,
                indent=2,
            )
        return path


query_logger = QueryLogger()


def log_request(question: str) -> None:
    logger.info("incoming question: %s", question[:120])


def log_error(error: str) -> None:
    logger.error("error: %s", error)