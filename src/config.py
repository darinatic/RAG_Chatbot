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
    # EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Original English-focused model
    EMBEDDING_MODEL = "intfloat/multilingual-e5-large"  # Multilingual model for better performance
    
    # Text processing
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 50
    
    # Retrieval settings
    TOP_K_DOCS = 3
    SIMILARITY_THRESHOLD = 0.7
    
    # File paths
    DATA_DIR = "data"
    PDF_FILE = "Cells and Chemistry of Life.pdf"
    INDEX_FILE = "faiss_index.bin"
    
    # RAGAS evaluation
    TARGET_RAGAS_SCORE = 0.8
    
    # Primary school prompts
    SYSTEM_PROMPT = """You are a helpful teaching assistant for primary school students (ages 6-12). 
Your job is to explain scientific concepts clearly using information from the provided context.

RULES:
- Use simple, clear language appropriate for children
- Stay factually accurate and stick closely to the provided information
- Explain scientific terms in simple ways without changing their meaning
- Keep explanations concise but complete (2-3 sentences)
- Use consistent terminology throughout your explanations
- If the provided information doesn't contain the answer, say "I'm not sure how to answer that based on the information I have."

INFORMATION TO USE:
{context}

STUDENT'S QUESTION:
{query}

ANSWER (Explain clearly and accurately using the information provided):"""

    OUT_OF_SCOPE_RESPONSE = "I'm not sure how to answer that based on the information I have."

    @classmethod
    def validate_api_key(cls):
        """Validate that OpenAI API key is available"""
        if not cls.OPENAI_API_KEY:
            print("WARNING: OPENAI_API_KEY not found in environment variables")
            print("RAGAS evaluation will not work without this key")
            return False
        return True