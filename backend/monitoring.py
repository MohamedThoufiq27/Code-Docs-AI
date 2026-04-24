import json
import time
from datetime import datetime
from pathlib import Path

LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(exist_ok=True)

# Token counting (approximate)
TOKENS_PER_1K_CHARS = 250
OPENROUTER_COST_PER_1M_INPUT = 0.0005  # $ for GPT-3.5-turbo
OPENROUTER_COST_PER_1M_OUTPUT = 0.0015

class QueryLogger:
    def __init__(self):
        self.queries = []
        self.total_cost = 0.0
    
    def log_query(self, question: str, latency: float, docs_retrieved: int):
        """Log query metadata"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "question": question[:100],  # First 100 chars
            "latency_ms": latency,
            "docs_retrieved": docs_retrieved
        }
        self.queries.append(log_entry)
        
        # Print to console
        print(f"📊 Query logged - Latency: {latency*1000:.2f}ms, Docs: {docs_retrieved}")
    
    def log_cost(self, prompt: str, response: str):
        """Calculate and log LLM costs"""
        # Approximate token counting
        input_tokens = len(prompt) / 4  # Rough estimate
        output_tokens = len(response) / 4
        
        input_cost = (input_tokens / 1_000_000) * OPENROUTER_COST_PER_1M_INPUT
        output_cost = (output_tokens / 1_000_000) * OPENROUTER_COST_PER_1M_OUTPUT
        total_cost = input_cost + output_cost
        
        self.total_cost += total_cost
        
        print(f"💰 Cost: ${total_cost:.6f} (Total: ${self.total_cost:.4f})")
    
    def save_logs(self):
        """Save logs to file"""
        log_file = LOGS_DIR / f"queries_{datetime.now().strftime('%Y%m%d')}.json"
        with open(log_file, 'w') as f:
            json.dump({
                "queries": self.queries,
                "total_cost": self.total_cost,
                "timestamp": datetime.now().isoformat()
            }, f, indent=2)

# Global logger instance
logger = QueryLogger()

def log_query(question: str, latency: float, docs_retrieved: int):
    logger.log_query(question, latency, docs_retrieved)

def log_cost(prompt: str, response: str):
    logger.log_cost(prompt, response)

def log_request(question: str):
    print(f"🔍 Incoming query: {question[:80]}...")

def log_error(error: str):
    print(f"❌ Error: {error}")

def get_total_cost():
    return logger.total_cost
