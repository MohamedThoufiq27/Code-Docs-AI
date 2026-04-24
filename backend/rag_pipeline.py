from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from .vector_store import VectorStore
from .config import Config
from .monitoring import log_query, log_cost
import json
import time

class RAGPipeline:
    def __init__(self):
        self.vectorstore = VectorStore()
        self.retriever = self.vectorstore.get_vectorstore().as_retriever(k=5)
        
        # Initialize LLM with OpenRouter
        self.llm = ChatOpenAI(
            model="openrouter/openai/gpt-3.5-turbo",
            openai_api_key=Config.OPENROUTER_API_KEY,
            openai_api_base="https://openrouter.ai/api/v1",
            temperature=0.3,  # Lower temp for consistent code suggestions
            max_tokens=1000
        )
        
        self.query_cache = {}  # Simple in-memory cache
        self._setup_prompt()
    
    def _setup_prompt(self):
        """Define RAG prompt template"""
        self.prompt = ChatPromptTemplate.from_template("""
You are an expert code documentation assistant. 

Based on the provided documentation and context, answer the user's question with:
1. A direct, concise answer
2. A code example if applicable
3. Key points to remember

Keep responses production-ready and focused.

Documentation Context:
{context}

User Question: {question}

Answer (respond in JSON format with fields: "answer", "code_example", "key_points"):
""")
    
    def _format_context(self, documents: list) -> str:
        """Format retrieved documents for context"""
        context = "\n\n---\n\n".join([
            f"Source: {doc.metadata.get('source', 'Unknown')}\n{doc.page_content}"
            for doc in documents
        ])
        return context[:Config.MAX_CONTEXT_LENGTH]  # Token-aware truncation
    
    def _parse_response(self, response: str) -> dict:
        """Parse LLM response as JSON"""
        try:
            # Extract JSON from response
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            return {"answer": response, "code_example": "", "key_points": []}
        except:
            return {"answer": response, "code_example": "", "key_points": []}
    
    def query(self, question: str, use_cache: bool = True):
        """Execute RAG query with cost tracking"""
        start_time = time.time()
        
        # Check cache
        if use_cache and question in self.query_cache:
            print("✅ Cache hit!")
            return self.query_cache[question]
        
        # Retrieve relevant documents
        print(f"🔍 Retrieving documents for: {question}")
        retrieved_docs = self.retriever.get_relevant_documents(question)
        context = self._format_context(retrieved_docs)
        
        # Format and execute RAG chain
        rag_chain = (
            {"context": lambda x: context, "question": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )
        
        response = rag_chain.invoke(question)
        parsed_response = self._parse_response(response)
        
        # Calculate metrics
        latency = time.time() - start_time
        
        # Log query and cost
        log_query(question, latency, len(retrieved_docs))
        log_cost(question, response)  # Track tokens/cost
        
        # Cache result
        if Config.CACHE_ENABLED:
            self.query_cache[question] = parsed_response
        
        return {
            "response": parsed_response,
            "sources": [doc.metadata.get('source') for doc in retrieved_docs],
            "latency_ms": latency * 1000,
            "cached": False
        }

# Example usage
if __name__ == "__main__":
    rag = RAGPipeline()
    
    # Test queries
    questions = [
        "How do I create a REST API in FastAPI?",
        "What is RAG in machine learning?",
        "How do I use async/await in Python?"
    ]
    
    for q in questions:
        result = rag.query(q)
        print(f"\n❓ Question: {q}")
        print(f"✅ Answer: {result['response']['answer']}")
        print(f"⏱️  Latency: {result['latency_ms']:.2f}ms")
        print(f"📚 Sources: {result['sources']}")
