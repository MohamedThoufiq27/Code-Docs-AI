import sys
import os

# Add parent directory to path so we can import from backend
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.rag_pipeline import RAGPipeline

def test_rag():
    print("⏳ Initializing RAG Pipeline testing...")
    rag = RAGPipeline()
    
    questions = [
        "How do I create a REST API in FastAPI?",
        "What is RAG in machine learning?"
    ]
    
    print("\\n=== Starting Queries ===")
    for q in questions:
        result = rag.query(q)
        print(f"\\n❓ Question: {q}")
        print(f"✅ Answer: {result['response'].get('answer', 'No Answer Found')}")
        if result['response'].get('code_example'):
            print(f"💻 Code: \\n{result['response'].get('code_example')}")
        print(f"⏱️  Latency: {result['latency_ms']:.2f}ms")
        print(f"📚 Sources: {result['sources']}")

if __name__ == "__main__":
    test_rag()
