from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.schema import Document
from .config import Config
import pinecone

class VectorStore:
    def __init__(self):
        # Initialize Pinecone
        pinecone.init(
            api_key=Config.PINECONE_API_KEY,
            environment=Config.PINECONE_ENV
        )
        
        # Initialize embeddings
        self.embeddings = OpenAIEmbeddings(
            openai_api_key=Config.OPENAI_API_KEY,
            model="text-embedding-3-small"  # Cost-effective model
        )
        
        # Get or create index
        self.index_name = Config.PINECONE_INDEX_NAME
        self._ensure_index_exists()
    
    def _ensure_index_exists(self):
        """Create index if it doesn't exist"""
        if self.index_name not in pinecone.list_indexes():
            print(f"Creating index: {self.index_name}")
            pinecone.create_index(
                name=self.index_name,
                dimension=1536,  # OpenAI embedding dimension
                metric="cosine"
            )
    
    def add_documents(self, chunks: list):
        """Vectorize and store documents in Pinecone"""
        print(f"🔄 Vectorizing and storing {len(chunks)} chunks...")
        
        vectorstore = Pinecone.from_documents(
            chunks,
            self.embeddings,
            index_name=self.index_name
        )
        
        print(f"✅ Successfully stored {len(chunks)} chunks in Pinecone")
        return vectorstore
    
    def get_vectorstore(self):
        """Get existing Pinecone vectorstore"""
        vectorstore = Pinecone.from_existing_index(
            self.index_name,
            self.embeddings
        )
        return vectorstore
    
    def similarity_search(self, query: str, k: int = 5):
        """Search for similar documents"""
        vectorstore = self.get_vectorstore()
        results = vectorstore.similarity_search(query, k=k)
        return results
