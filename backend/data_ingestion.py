from langchain.document_loaders import DirectoryLoader, WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from .config import Config

class DocumentIngestion:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP,
            separators=["\n\n", "\n", " ", ""]
        )
    
    def load_local_docs(self, docs_path: str):
        """Load documentation from local directory"""
        loader = DirectoryLoader(docs_path, glob="**/*.md")
        documents = loader.load()
        return documents
    
    def load_web_docs(self, urls: list):
        """Load documentation from web URLs"""
        documents = []
        for url in urls:
            loader = WebBaseLoader(url)
            docs = loader.load()
            documents.extend(docs)
        return documents
    
    def chunk_documents(self, documents: list):
        """Split documents into chunks"""
        chunks = self.text_splitter.split_documents(documents)
        print(f"✅ Created {len(chunks)} chunks from {len(documents)} documents")
        return chunks
    
    def prepare_data(self, docs_path: str, web_urls: list = None):
        """Full pipeline: load → chunk → return"""
        # Load local docs
        local_docs = self.load_local_docs(docs_path)
        
        # Load web docs if provided
        if web_urls:
            web_docs = self.load_web_docs(web_urls)
            all_docs = local_docs + web_docs
        else:
            all_docs = local_docs
        
        # Chunk documents
        chunks = self.chunk_documents(all_docs)
        return chunks
