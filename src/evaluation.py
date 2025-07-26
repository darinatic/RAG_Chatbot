"""
RAGAS evaluation for RAG chatbot with MLflow tracking
"""
import os
import numpy as np
from typing import List, Dict, Any
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from .rag_chatbot import RAGChatbot
from .config import Config
import mlflow
from datetime import datetime

class RAGASEvaluator:
    """Evaluate RAG system using RAGAS metrics with MLflow tracking"""
    
    def __init__(self, chatbot: RAGChatbot):
        self.chatbot = chatbot
        self.config = Config()
        
        # Set OpenAI API key in environment if available from config
        if self.config.OPENAI_API_KEY:
            os.environ["OPENAI_API_KEY"] = self.config.OPENAI_API_KEY
        else:
            raise ValueError(
                "OpenAI API key not found. Please add OPENAI_API_KEY to your .env file"
            )
    
    def create_test_dataset(self, test_cases: List[Dict[str, str]]) -> Dataset:
        """Create evaluation dataset for RAGAS
        
        Args:
            test_cases: List of dicts with 'question' and 'ground_truth' keys
        """
        evaluation_data = {
            'question': [],
            'answer': [],
            'contexts': [],
            'ground_truth': []
        }
        
        for case in test_cases:
            query = case['question']
            ground_truth = case['ground_truth']
            
            # Get chatbot response
            result = self.chatbot.answer(query)
            
            # Get contexts from vector store
            retrieved_chunks = self.chatbot.vector_store.search(query)
            contexts = [chunk['content'] for chunk in retrieved_chunks]
            
            evaluation_data['question'].append(query)
            evaluation_data['answer'].append(result['answer'])
            evaluation_data['contexts'].append(contexts)
            evaluation_data['ground_truth'].append(ground_truth)
        
        return Dataset.from_dict(evaluation_data)
    
    def _process_ragas_result(self, value):
        """Process RAGAS result - handle both list and single values"""
        if isinstance(value, list):
            return float(np.mean(value))
        else:
            return float(value)
    
    def evaluate_with_mlflow(self, test_cases: List[Dict[str, str]], chunks_count: int = 0) -> Dict[str, float]:
        """Run RAGAS evaluation with MLflow tracking"""
        
        # Create descriptive run name
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_name = f"{self.config.EMBEDDING_MODEL}_chunk{self.config.CHUNK_SIZE}_top{self.config.TOP_K_DOCS}_sim{self.config.SIMILARITY_THRESHOLD}_{timestamp}"
        
        # Start MLflow run
        with mlflow.start_run(run_name=run_name):
            # Log configuration parameters
            mlflow.log_param("llm_model", self.config.OLLAMA_MODEL)
            mlflow.log_param("embedding_model", self.config.EMBEDDING_MODEL)
            mlflow.log_param("chunk_size", self.config.CHUNK_SIZE)
            mlflow.log_param("chunk_overlap", self.config.CHUNK_OVERLAP)
            mlflow.log_param("top_k_docs", self.config.TOP_K_DOCS)
            mlflow.log_param("similarity_threshold", self.config.SIMILARITY_THRESHOLD)
            mlflow.log_param("total_chunks", chunks_count)
            mlflow.log_param("test_cases_count", len(test_cases))
            
            try:
                # Run evaluation
                scores = self.evaluate(test_cases)
                
                if 'error' not in scores:
                    # Log RAGAS metrics
                    mlflow.log_metric("faithfulness", scores['faithfulness'])
                    mlflow.log_metric("answer_relevancy", scores['answer_relevancy'])
                    mlflow.log_metric("context_precision", scores['context_precision'])
                    mlflow.log_metric("context_recall", scores['context_recall'])
                    mlflow.log_metric("overall_score", scores['overall_score'])
                    
                    # Log whether target was achieved
                    target_achieved = scores['overall_score'] >= self.config.TARGET_RAGAS_SCORE
                    mlflow.log_metric("target_achieved", 1 if target_achieved else 0)
                    
                    # Add experiment notes
                    notes = f"RAGAS evaluation run on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                    mlflow.set_tag("notes", notes)
                    mlflow.set_tag("pdf_file", self.config.PDF_FILE)
                    
                    print("MLflow tracking successful!")
                    
                else:
                    mlflow.log_param("evaluation_error", scores['error'])
                    print(f"Evaluation error: {scores['error']}")
                
                return scores
                
            except Exception as e:
                mlflow.log_param("evaluation_exception", str(e))
                print(f"RAGAS evaluation failed: {str(e)}")
                return {'error': str(e)}
    
    def evaluate(self, test_cases: List[Dict[str, str]]) -> Dict[str, float]:
        """Run RAGAS evaluation"""
        try:
            # Verify API key is set
            if not os.getenv("OPENAI_API_KEY"):
                return {'error': 'OpenAI API key not found in environment variables'}
            
            test_dataset = self.create_test_dataset(test_cases)
            
            results = evaluate(
                dataset=test_dataset,
                metrics=[
                    faithfulness,
                    answer_relevancy,
                    context_precision,
                    context_recall
                ]
            )
            
            # Handle both list and single value results from RAGAS 0.3.0
            scores = {
                'faithfulness': self._process_ragas_result(results['faithfulness']),
                'answer_relevancy': self._process_ragas_result(results['answer_relevancy']),
                'context_precision': self._process_ragas_result(results['context_precision']),
                'context_recall': self._process_ragas_result(results['context_recall'])
            }
            
            scores['overall_score'] = sum(scores.values()) / len(scores)
            
            return scores
            
        except Exception as e:
            print(f"RAGAS evaluation error: {str(e)}")
            return {'error': str(e)}
    
    def print_evaluation_report(self, scores: Dict[str, float], target_score: float = 0.8):
        """Print formatted evaluation report"""
        print("\n" + "="*50)
        print("RAGAS EVALUATION REPORT")
        print("="*50)
        
        if 'error' in scores:
            print(f"Evaluation failed: {scores['error']}")
            return
        
        for metric, score in scores.items():
            if metric != 'overall_score':
                status = "PASS" if score >= target_score else "FAIL"
                print(f"{status} {metric}: {score:.3f}")
        
        overall = scores['overall_score']
        overall_status = "PASS" if overall >= target_score else "FAIL"
        print(f"\n{overall_status} Overall Score: {overall:.3f}")
        print(f"Target Score: {target_score}")
        
        if overall >= target_score:
            print("Target achieved! System ready for deployment.")
        else:
            print("Target not reached. Consider:")
            print("   - Better quality prompts")
            print("   - Improved chunking strategy")
            print("   - Different retrieval parameters")
    
    @staticmethod
    def create_sample_test_cases() -> List[Dict[str, str]]:
        """Create sample test cases for cells and chemistry of life"""
        return [
            {
                "question": "What Are Cells?",
                "ground_truth": "Cells are the building blocks of life. They are tiny structures that carry out all the activities needed to keep living things alive, like taking in materials, making useful molecules, and giving energy. You can think of cells like little factories inside your body."
            },
            {
                "question": "What Are Enzymes?",
                "ground_truth": "Enzymes are special molecules that help break down or join other molecules together during chemical reactions in the body. For example, an enzyme called maltase breaks down maltose into smaller sugar units like glucose. Enzymes make these reactions happen faster and more easily."
            },
            {
                "question": "What Is Osmosis?",
                "ground_truth": "Osmosis is the movement of water from a place with more water to a place with less water, through a special wall that only lets some things through. This helps cells take in water or release it, depending on their surroundings."
            },
            {
                "question": "What Are Complex Carbohydrates?",
                "ground_truth": "Complex carbohydrates are large molecules made from many sugar units joined together. Examples include starch, cellulose, and glycogen, which have different structures and roles, like storing energy or building cell walls in plants."
            },
            {
                "question": "What Are Proteins?",
                "ground_truth": "Proteins are important molecules made by the cell to help build body parts and do special jobs. Cells use tiny machines called ribosomes to make proteins, which help with growth, repair, and many body functions."
            }
        ]