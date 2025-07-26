"""
Main RAG Chatbot implementation with language detection
"""
import ollama
import requests
from langdetect import detect, DetectorFactory
from typing import Dict, Any, List, Optional
from .config import Config
from .vector_store import VectorStore

# Set seed for consistent language detection
DetectorFactory.seed = 0

class RAGChatbot:
    """Main RAG chatbot with local LLM and language detection"""
    
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
    
    def _detect_language(self, text: str) -> str:
        """Detect the language of input text"""
        try:
            lang_code = detect(text)
            
            # Map language codes to readable names
            lang_mapping = {
                'en': 'English',
                'zh': 'Chinese',  # Covers both simplified and traditional
                'zh-cn': 'Chinese',
                'zh-tw': 'Chinese',
                'ms': 'Malay',
                'id': 'Malay',  # Indonesian is similar to Malay
            }
            
            return lang_mapping.get(lang_code, 'English')
        except:
            # Default to English if detection fails
            return 'English'
    
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
    
    def _create_prompt(self, query: str, context_chunks: List[Dict], detected_language: str) -> str:
        """Create prompt with context and language instruction for LLM"""
        context = "\n".join([chunk['content'] for chunk in context_chunks])
        
        # Create language-aware system prompt
        language_instruction = f"\nIMPORTANT: The student asked the question in {detected_language}. You must respond in {detected_language}."
        
        system_prompt_with_language = self.config.SYSTEM_PROMPT + language_instruction
        
        return system_prompt_with_language.format(
            context=context,
            query=query
        )
    
    def answer(self, query: str) -> Dict[str, Any]:
        """Main method to answer user queries with language detection"""
        # Step 1: Detect language of the query
        detected_language = self._detect_language(query)
        
        # Step 2: Retrieve relevant chunks
        retrieved_chunks = self.vector_store.search(query)
        
        # Step 3: Check if query is in scope
        if not self._is_in_scope(retrieved_chunks):
            # Create language-specific out-of-scope response
            if detected_language == 'Chinese':
                out_of_scope_response = "我不确定如何根据我拥有的信息回答这个问题。"
            elif detected_language == 'Malay':
                out_of_scope_response = "Saya tidak pasti bagaimana untuk menjawab soalan itu berdasarkan maklumat yang saya ada."
            else:
                out_of_scope_response = self.config.OUT_OF_SCOPE_RESPONSE
            
            return {
                'query': query,
                'answer': out_of_scope_response,
                'detected_language': detected_language,
                'in_scope': False,
                'confidence': 0.0,
                'chunks_used': 0
            }
        
        # Step 4: Generate answer using RAG with language awareness
        prompt = self._create_prompt(query, retrieved_chunks, detected_language)
        answer = self._query_llm(prompt)
        
        # Step 5: Calculate confidence
        avg_similarity = sum([chunk.get('similarity', 0) for chunk in retrieved_chunks]) / len(retrieved_chunks)
        
        # Store conversation
        result = {
            'query': query,
            'answer': answer,
            'detected_language': detected_language,
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