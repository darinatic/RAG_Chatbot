"""
Document processing for PDF knowledge base
"""
import os
import re
from typing import List, Dict, Any
import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter
from .config import Config

class DocumentProcessor:
    """Handle PDF loading, cleaning, and chunking"""
    
    def __init__(self, config: Config = Config()):
        self.config = config
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
    
    def load_pdf(self, pdf_path: str = None) -> str:
        """Load and extract text from PDF"""
        if pdf_path is None:
            pdf_path = os.path.join(self.config.DATA_DIR, self.config.PDF_FILE)
        
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
                
                print(f"Successfully loaded PDF: {len(text)} characters")
                return self.clean_text(text)
                
        except Exception as e:
            print(f"Error loading PDF: {str(e)}")
            return ""
    
    def clean_text(self, text: str) -> str:
        """Clean and preprocess extracted text"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove excessive punctuation
        text = re.sub(r'[.]{3,}', '...', text)
        
        # Normalize quotes
        text = re.sub(r'["""]', '"', text)
        #text = re.sub(r'[''']', "'", text)
        
        return text.strip()
    
    def chunk_text(self, text: str) -> List[Dict[str, Any]]:
        """Split text into chunks with metadata"""
        chunks = self.text_splitter.split_text(text)
        
        chunked_docs = []
        for i, chunk in enumerate(chunks):
            chunked_docs.append({
                'content': chunk,
                'chunk_id': i,
                'metadata': {
                    'source': self.config.PDF_FILE,
                    'chunk_index': i
                }
            })
        
        print(f"Created {len(chunked_docs)} chunks")
        return chunked_docs
    
    def process_document(self, pdf_path: str = None) -> List[Dict[str, Any]]:
        """Complete document processing pipeline"""
        text = self.load_pdf(pdf_path)
        if not text:
            return []
        
        return self.chunk_text(text)