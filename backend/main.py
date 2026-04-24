from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from .rag_pipeline import RAGPipeline
from .monitoring import log_request, log_error
from .config import Config
import time

app = FastAPI(title=Config.APP_NAME, debug=Config.DEBUG)

# CORS setup for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize RAG pipeline
rag_pipeline = RAGPipeline()

# Request/Response models
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
    """Health check endpoint"""
    return {"status": "healthy", "service": Config.APP_NAME}

@app.post("/query", response_model=QueryResponse)
async def query_docs(request: QueryRequest):
    """Main RAG query endpoint"""
    try:
        log_request(request.question)
        
        # Execute RAG pipeline
        result = rag_pipeline.query(request.question, request.use_cache)
        
        return QueryResponse(**result)
    
    except Exception as e:
        log_error(str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats")
async def get_stats():
    """Get query statistics and costs"""
    return {
        "total_queries": len(rag_pipeline.query_cache),
        "cache_size": len(rag_pipeline.query_cache),
        "index_name": Config.PINECONE_INDEX_NAME
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
