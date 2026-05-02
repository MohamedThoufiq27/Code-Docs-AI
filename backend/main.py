from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from .config import Config
from .monitoring import log_error, log_request, query_logger
from .rag_pipeline import RAGPipeline

limiter = Limiter(key_func=get_remote_address)

app = FastAPI(title=Config.APP_NAME, debug=Config.DEBUG)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lazy: only build the pipeline when a real query arrives. /health must work
# without API keys so smoke tests pass on a fresh checkout.
_rag_pipeline: RAGPipeline | None = None


def get_pipeline() -> RAGPipeline:
    global _rag_pipeline
    if _rag_pipeline is None:
        _rag_pipeline = RAGPipeline()
    return _rag_pipeline


class QueryRequest(BaseModel):
    question: str
    use_cache: bool = True


class QueryResponse(BaseModel):
    response: dict
    sources: list
    latency_ms: float
    cached: bool


@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": Config.APP_NAME}


@app.post("/query", response_model=QueryResponse)
@limiter.limit(Config.RATE_LIMIT)
async def query_docs(request: Request, body: QueryRequest):
    try:
        log_request(body.question)
        result = get_pipeline().query(body.question, body.use_cache)
        return QueryResponse(**result)
    except Exception as e:
        log_error(str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats")
async def get_stats():
    pipeline = get_pipeline()
    try:
        stats = pipeline.vector_store.index_stats()
    except Exception as e:
        stats = {"error": str(e)}
    return {
        "index_name": Config.PINECONE_INDEX_NAME,
        "index_stats": stats,
        "total_queries": query_logger.total_queries,
        "total_cost_usd": query_logger.total_cost,
        "cache_size": len(pipeline.query_cache),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)