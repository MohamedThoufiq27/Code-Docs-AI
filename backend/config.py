import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # LLM Configuration
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # For embeddings
    
    # Vector DB Configuration
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    PINECONE_ENV = os.getenv("PINECONE_ENV", "gcp-starter")
    PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "codedocs-ai")
    
    # FastAPI Configuration
    APP_NAME = "CodeDocsAI"
    DEBUG = os.getenv("DEBUG", "False") == "True"
    
    # Cost Optimization
    CHUNK_SIZE = 500  # Characters per chunk
    CHUNK_OVERLAP = 50
    MAX_CONTEXT_LENGTH = 3000  # Token limit for context
    CACHE_ENABLED = True
    CACHE_TTL = 3600  # 1 hour
