"""
Streamlit Chat Interface for RAG Chatbot
Kid-friendly interface for primary school students
"""

import streamlit as st
import sys
import os
from pathlib import Path

# Add src to path
sys.path.append('src')

from src import Config, DocumentProcessor, VectorStore, RAGChatbot
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Science Helper Bot 🔬",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

def load_css():
    """Load custom CSS styling"""
    try:
        with open('static/style.css') as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    except FileNotFoundError:
        # Fallback to inline CSS if file doesn't exist
        st.markdown("""
        <style>
            .main { padding-top: 2rem; }
            .stApp { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }
            .title-container {
                background: linear-gradient(90deg, #FF6B6B, #4ECDC4, #45B7D1, #96CEB4);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
                font-size: 3rem;
                font-weight: bold;
                text-align: center;
                margin-bottom: 1rem;
            }
            .subtitle {
                color: white;
                text-align: center;
                font-size: 1.2rem;
                margin-bottom: 2rem;
            }
            .user-message {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 15px;
                border-radius: 15px;
                margin: 10px 0;
                box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
            }
            .bot-message {
                background: linear-gradient(135deg, #4ECDC4 0%, #44A08D 100%);
                color: white;
                padding: 15px;
                border-radius: 15px;
                margin: 10px 0;
                box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
            }
            .confidence-badge {
                background: rgba(255, 255, 255, 0.2);
                padding: 5px 10px;
                border-radius: 20px;
                font-size: 0.8rem;
                margin-top: 10px;
                display: inline-block;
            }
        </style>
        """, unsafe_allow_html=True)

# Load styling
load_css()

@st.cache_resource
def initialize_chatbot():
    """Initialize the chatbot components (cached for performance)"""
    try:
        config = Config()
        
        # Initialize document processor
        doc_processor = DocumentProcessor(config)
        
        # Initialize vector store
        vector_store = VectorStore(config)
        
        # Try to load existing index, otherwise create new one
        if not vector_store.load_index():
            st.info("🔄 Setting up the knowledge base for the first time... This might take a moment!")
            chunks = doc_processor.process_document()
            if not chunks:
                st.error("❌ Could not load the PDF file. Please check if 'Cells and Chemistry of Life.pdf' exists in the data/ directory.")
                return None
            vector_store.setup_from_chunks(chunks)
            vector_store.save_index()
        
        # Initialize chatbot
        chatbot = RAGChatbot(vector_store, config)
        
        return chatbot
    
    except Exception as e:
        st.error(f"❌ Failed to initialize chatbot: {str(e)}")
        st.error("Make sure Ollama is running and the gemma3:4b model is available.")
        return None

def display_message(message, is_user=False):
    """Display a chat message with kid-friendly styling"""
    if is_user:
        with st.container():
            st.markdown(f"""
            <div class="user-message">
                <strong>You asked:</strong> {message}
            </div>
            """, unsafe_allow_html=True)
    else:
        with st.container():
            st.markdown(f"""
            <div class="bot-message">
                <strong>🤖 Science Helper:</strong> {message['answer']}
                <div class="confidence-badge">
                    Confidence: {message['confidence']:.0%} | Sources: {message['chunks_used']}
                </div>
            </div>
            """, unsafe_allow_html=True)

def main():
    # Title and subtitle
    st.markdown("""
    <div class="title-container">
        🔬 Science Helper Bot 🔬
    </div>
    <div class="subtitle">
        Ask me anything about cells and chemistry! I'm here to help you learn! 🌟
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize chatbot
    chatbot = initialize_chatbot()
    
    if chatbot is None:
        st.stop()
    
    # Initialize session state for chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
        # Add welcome message
        welcome_msg = {
            'answer': "Hi there! 👋 I'm your Science Helper Bot! I can answer questions about cells and chemistry. Try asking me something like 'What is a cell?' or 'What do mitochondria do?'",
            'confidence': 1.0,
            'chunks_used': 0
        }
        st.session_state.messages.append(("bot", welcome_msg))
    
    # Display chat history
    chat_container = st.container()
    with chat_container:
        for role, message in st.session_state.messages:
            if role == "user":
                display_message(message, is_user=True)
            else:
                display_message(message, is_user=False)
    
    # Chat input
    if prompt := st.chat_input("Ask me about cells and chemistry! 🧪"):
        # Add user message to history
        st.session_state.messages.append(("user", prompt))
        
        # Display user message immediately
        with chat_container:
            display_message(prompt, is_user=True)
        
        # Show thinking indicator
        with st.spinner("🤔 Let me think about that..."):
            # Get chatbot response
            try:
                response = chatbot.answer(prompt)
                
                # Add bot response to history
                st.session_state.messages.append(("bot", response))
                
                # Display bot response
                with chat_container:
                    display_message(response, is_user=False)
                
                # Rerun to update the display
                st.rerun()
                
            except Exception as e:
                error_response = {
                    'answer': "Oops! I'm having trouble thinking right now. Can you try asking your question again? 😅",
                    'confidence': 0.0,
                    'chunks_used': 0
                }
                st.session_state.messages.append(("bot", error_response))
                st.rerun()
    
    # Clear chat button in sidebar (hidden by default)
    with st.sidebar:
        st.markdown("### Chat Options")
        if st.button("🗑️ Clear Chat", help="Start a new conversation"):
            st.session_state.messages = []
            # Re-add welcome message
            welcome_msg = {
                'answer': "Hi there! 👋 I'm your Science Helper Bot! I can answer questions about cells and chemistry. Try asking me something like 'What is a cell?' or 'What do mitochondria do?'",
                'confidence': 1.0,
                'chunks_used': 0
            }
            st.session_state.messages.append(("bot", welcome_msg))
            st.rerun()
        
        # Show some example questions
        st.markdown("### Try asking:")
        st.markdown("""
        - What is a cell?
        - What do mitochondria do?
        - How do red blood cells work?
        - What is diffusion?
        - What's inside a plant cell?
        """)
        
        # System info (for debugging)
        if st.checkbox("Show system info"):
            st.markdown("### System Status")
            st.success("✅ Chatbot loaded")
            st.info(f"📊 Total messages: {len(st.session_state.messages)}")

if __name__ == "__main__":
    main()