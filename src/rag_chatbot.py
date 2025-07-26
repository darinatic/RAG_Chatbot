"""
Main RAG Chatbot implementation
"""
import ollama
import requests
from typing import Dict, Any, List, Optional
from .config import Config
from .vector_store import VectorStore

class RAGChatbot:
    """Main RAG chatbot with local LLM"""
    
    def __init__(self, vector_store: VectorStore, config: Config = Config()):
        self.vector_store = vector_store
        self.config = config
        self.conversation_history = []
        
        # Test Ollama connection
        if not self._test_ollama():
            raise ConnectionError("Cannot connect to Ollama. Make sure it's running.")
    
    def _test_ollama(self) -> bool:
        """Test Ollama connection and model availability"""
        try:
            response = requests.get(f"{self.config.OLLAMA_BASE_URL}/api/tags")
            if response.status_code == 200:
                models = response.json()
                model_names = [model['name'] for model in models.get('models', [])]
                
                if self.config.OLLAMA_MODEL in model_names:
                    print(f"Ollama connected - {self.config.OLLAMA_MODEL} available")
                    return True
                else:
                    print(f"Model {self.config.OLLAMA_MODEL} not found")
                    return False
            return False
        except Exception as e:
            print(f"Ollama connection error: {str(e)}")
            return False
    
    def _query_llm(self, prompt: str) -> str:
        """Query local Ollama model"""
        try:
            response = ollama.chat(
                model=self.config.OLLAMA_MODEL,
                messages=[{'role': 'user', 'content': prompt}],
                options={'temperature': 0.7}
            )
            return response['message']['content'].strip()
        except Exception as e:
            print(f"LLM query error: {str(e)}")
            return "I'm having trouble thinking right now. Can you try asking again?"
    
    def _is_in_scope(self, retrieved_chunks: List[Dict]) -> bool:
        """Check if query can be answered from retrieved chunks"""
        if not retrieved_chunks:
            return False
        
        # Check if best match meets similarity threshold
        best_similarity = max([chunk.get('similarity', 0) for chunk in retrieved_chunks])
        return best_similarity > self.config.SIMILARITY_THRESHOLD
    
    def _create_prompt(self, query: str, context_chunks: List[Dict]) -> str:
        """Create prompt with context for LLM"""
        context = "\n".join([chunk['content'] for chunk in context_chunks])
        
        return self.config.SYSTEM_PROMPT.format(
            context=context,
            query=query
        )
    
    def answer(self, query: str) -> Dict[str, Any]:
        """Main method to answer user queries"""
        # Step 1: Retrieve relevant chunks
        retrieved_chunks = self.vector_store.search(query)
        
        # Step 2: Check if query is in scope
        if not self._is_in_scope(retrieved_chunks):
            return {
                'query': query,
                'answer': self.config.OUT_OF_SCOPE_RESPONSE,
                'in_scope': False,
                'confidence': 0.0,
                'chunks_used': 0
            }
        
        # Step 3: Generate answer using RAG
        prompt = self._create_prompt(query, retrieved_chunks)
        answer = self._query_llm(prompt)
        
        # Step 4: Calculate confidence
        avg_similarity = sum([chunk.get('similarity', 0) for chunk in retrieved_chunks]) / len(retrieved_chunks)
        
        # Store conversation
        result = {
            'query': query,
            'answer': answer,
            'in_scope': True,
            'confidence': float(avg_similarity),
            'chunks_used': len(retrieved_chunks)
        }
        
        self.conversation_history.append(result)
        return result
    
    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """Get conversation history"""
        return self.conversation_history
    
    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []