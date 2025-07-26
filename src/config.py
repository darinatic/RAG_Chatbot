"""
Configuration settings for RAG Chatbot
"""
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    # Ollama settings
    OLLAMA_MODEL = "gemma3:4b"
    OLLAMA_BASE_URL = "http://localhost:11434"
    
    # OpenAI API Key 
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    
    # Embedding model
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    
    # Text processing
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 50
    
    # Retrieval settings
    TOP_K_DOCS = 5
    SIMILARITY_THRESHOLD = 0.5
    
    # File paths
    DATA_DIR = "data"
    PDF_FILE = "Cells and Chemistry of Life.pdf"
    INDEX_FILE = "faiss_index.bin"
    
    # RAGAS evaluation
    TARGET_RAGAS_SCORE = 0.8
    
    # Primary school prompts
    SYSTEM_PROMPT = """You are a helpful teaching assistant for primary school students (ages 6-12). 
Your job is to answer questions in a simple, clear, and engaging way.

RULES:
- Use simple words that kids can understand
- Be cheerful and encouraging
- Use examples that kids can relate to
- Keep answers short but complete (2-3 sentences)
- If you don't know something from the given information, say "I'm not sure how to answer that based on the information I have."

INFORMATION TO USE:
{context}

STUDENT'S QUESTION:
{query}

ANSWER (Keep it simple and clear for kids!):"""

    OUT_OF_SCOPE_RESPONSE = "I'm not sure how to answer that based on the information I have."

    @classmethod
    def validate_api_key(cls):
        """Validate that OpenAI API key is available"""
        if not cls.OPENAI_API_KEY:
            print("WARNING: OPENAI_API_KEY not found in environment variables")
            print("RAGAS evaluation will not work without this key")
            return False
        return True