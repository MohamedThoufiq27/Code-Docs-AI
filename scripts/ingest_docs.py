import sys
import os

# Add parent directory to path so we can import from backend
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.data_ingestion import DocumentIngestion
from backend.vector_store import VectorStore

def main():
    print("🚀 Starting Data Ingestion Pipeline...")
    
    # 1. Initialize ingestion
    ingestion = DocumentIngestion()
    
    # Define sample docs to ingest (you can modify this)
    docs_path = "./data/docs" # Local directory
    web_urls = [
        "https://docs.python.org/3/tutorial/index.html",
        "https://fastapi.tiangolo.com/"
    ]
    
    # Ensure local directory exists
    os.makedirs(docs_path, exist_ok=True)
    
    # Enable a simple README to be there for now
    readme_path = os.path.join(docs_path, "sample_doc.md")
    if not os.path.exists(readme_path):
        with open(readme_path, "w") as f:
            f.write("# Sample Doc\\nThis is a sample documentation file to start indexing. CodeDocsAI is cool!")
    
    # 2. Extract and chunk data
    chunks = ingestion.prepare_data(docs_path, web_urls)
    print(f"✅ Prepared {len(chunks)} document chunks.")
    
    # 3. Store in Vector DB
    if chunks:
        vector_store = VectorStore()
        vector_store.add_documents(chunks)
        print("✅ Data successfully ingested into Pinecone!")
    else:
        print("⚠️ No chunks found to ingest.")

if __name__ == "__main__":
    main()
