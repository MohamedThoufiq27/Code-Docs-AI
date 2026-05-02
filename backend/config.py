import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    # LLM / embeddings API keys
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # used only for embeddings

    # Models (overridable via .env)
    LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-3.5-turbo")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

    # Pinecone v5 (cloud + region replaces the old PINECONE_ENV)
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "codedocs-ai")
    PINECONE_CLOUD = os.getenv("PINECONE_CLOUD", "aws")
    PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")
    PINECONE_DIMENSION = 1536  # text-embedding-3-small

    # FastAPI
    APP_NAME = "CodeDocsAI"
    DEBUG = os.getenv("DEBUG", "False") == "True"

    # Chunking — units are TOKENS (cl100k_base via tiktoken), not characters.
    # Stage 2 of the splitter uses RecursiveCharacterTextSplitter.from_tiktoken_encoder.
    CHUNK_SIZE = 800
    CHUNK_OVERLAP = 100
    MAX_CONTEXT_LENGTH = 3000  # chars, for context truncation safety net

    # Retrieval
    INITIAL_K = 20  # vector search top-K (stage 1)
    FINAL_K = 5     # cross-encoder rerank top-K (stage 2)

    # Cache (cachetools.TTLCache)
    CACHE_ENABLED = True
    CACHE_TTL = 3600           # seconds
    CACHE_MAX_SIZE = 1000

    # Rate limiting (slowapi)
    RATE_LIMIT = os.getenv("RATE_LIMIT", "10/minute")

    # OpenRouter pricing for openai/gpt-3.5-turbo as of 2026-04
    # https://openrouter.ai/models  -- verify before relying on these in prod
    COST_PER_1M_INPUT = 0.50
    COST_PER_1M_OUTPUT = 1.50