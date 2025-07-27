"""
Streamlit Chat Interface for RAG Chatbot
Clean interface for primary school students
"""

import streamlit as st
import sys
#import os
#from pathlib import Path

# Add src to path
sys.path.append('src')

from src import Config, VectorStore, RAGChatbot
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Science Helper Bot",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def initialize_chatbot():
    """Initialize the chatbot components (cached for performance)"""
    try:
        config = Config()
        
        # Initialize vector store
        vector_store = VectorStore(config)
        
        # Load pre-built index (must exist in container)
        if not vector_store.load_index():
            st.error("Failed to load pre-built FAISS index. Please check if faiss_index.bin and faiss_index_chunks.pkl exist.")
            return None
        
        # Initialize chatbot
        chatbot = RAGChatbot(vector_store, config)
        
        return chatbot
    
    except Exception as e:
        st.error(f"Failed to initialize chatbot: {str(e)}")
        st.error("Make sure Ollama is running and the gemma3:4b model is available.")
        return None

def display_message(message, is_user=False):
    """Display a chat message"""
    if is_user:
        with st.chat_message("user"):
            st.write(message)
    else:
        with st.chat_message("assistant"):
            st.write(message['answer'])
            
            # Show confidence and source information
            col1, col2 = st.columns(2)
            with col1:
                st.caption(f"Confidence: {message['confidence']:.0%}")
            with col2:
                st.caption(f"Sources used: {message['chunks_used']}")

def main():
    # Title and description
    st.title("Science Helper Bot")
    st.markdown("Ask me anything about cells and chemistry! I'm here to help you learn.")
    
    # Initialize chatbot
    chatbot = initialize_chatbot()
    
    if chatbot is None:
        st.stop()
    
    # Initialize session state for chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
        # Add welcome message
        welcome_msg = {
            'answer': "Hi there! I'm your Science Helper Bot! I can answer questions about cells and chemistry. Try asking me something like 'What is a cell?' or 'What do mitochondria do?'",
            'confidence': 1.0,
            'chunks_used': 0
        }
        st.session_state.messages.append(("assistant", welcome_msg))
    
    # Display chat history
    for role, message in st.session_state.messages:
        if role == "user":
            display_message(message, is_user=True)
        else:
            display_message(message, is_user=False)
    
    # Chat input
    if prompt := st.chat_input("Ask me about cells and chemistry!"):
        # Add user message to history and display
        st.session_state.messages.append(("user", prompt))
        display_message(prompt, is_user=True)
        
        # Get chatbot response
        try:
            with st.spinner("Thinking..."):
                response = chatbot.answer(prompt)
                
                # Add bot response to history and display
                st.session_state.messages.append(("assistant", response))
                display_message(response, is_user=False)
                
        except Exception as e:
            error_response = {
                'answer': "I'm having trouble thinking right now. Can you try asking your question again?",
                'confidence': 0.0,
                'chunks_used': 0
            }
            st.session_state.messages.append(("assistant", error_response))
            display_message(error_response, is_user=False)
    
    # Sidebar with options and information
    with st.sidebar:
        st.header("Chat Options")
        
        # Clear chat button
        if st.button("Clear Chat", type="primary"):
            st.session_state.messages = []
            # Re-add welcome message
            welcome_msg = {
                'answer': "Hi there! I'm your Science Helper Bot! I can answer questions about cells and chemistry. Try asking me something like 'What is a cell?' or 'What do mitochondria do?'",
                'confidence': 1.0,
                'chunks_used': 0
            }
            st.session_state.messages.append(("assistant", welcome_msg))
            st.rerun()
        
        st.divider()
        
        # Example questions
        st.header("Example Questions")
        st.markdown("""
        Try asking me about:
        - What is a cell?
        - What do mitochondria do?
        - How do red blood cells work?
        - What is diffusion?
        - What's inside a plant cell?
        - What are enzymes?
        - What is osmosis?
        """)
        
        st.divider()
        
        # Language support information
        st.header("Language Support")
        st.info("I can answer questions in English, Chinese, and Malay!")
        
        st.divider()
        
        # System information
        with st.expander("System Information"):
            st.success("Chatbot: Online")
            st.info(f"Total messages: {len(st.session_state.messages)}")
            
            if st.session_state.messages:
                # Calculate average confidence
                confidences = [msg[1]['confidence'] for msg in st.session_state.messages if msg[0] == "assistant" and 'confidence' in msg[1]]
                if confidences:
                    avg_confidence = sum(confidences) / len(confidences)
                    st.metric("Average Confidence", f"{avg_confidence:.1%}")

if __name__ == "__main__":
    main()