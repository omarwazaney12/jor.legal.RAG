#!/usr/bin/env python3
"""
Deployment version of Advanced Legal RAG - works with existing ChromaDB only
This version doesn't require original documents, only the embedded database
"""

import os
from pathlib import Path
from advanced_rag_system import AdvancedLegalRAGSystem

class DeploymentRAGSystem(AdvancedLegalRAGSystem):
    """RAG system optimized for deployment - works with existing embeddings only"""
    
    def __init__(self, chroma_db_path: str = "./chroma_db"):
        self.chroma_db_path = Path(chroma_db_path)
        
        # Initialize without documents path
        from advanced_rag_system import AdvancedLegalProcessor, AdvancedVectorStore, LegalReasoningEngine
        
        self.processor = AdvancedLegalProcessor()
        self.vector_store = AdvancedVectorStore()
        self.reasoning_engine = LegalReasoningEngine(self.vector_store)
        self.documents = []  # Empty - we only use embeddings
        
        print("🚀 Deployment RAG System initialized (embeddings only)")
    
    def load_embeddings_only(self) -> bool:
        """Load existing ChromaDB embeddings without processing documents"""
        
        if not self.chroma_db_path.exists():
            print(f"❌ ChromaDB not found at: {self.chroma_db_path}")
            return False
        
        try:
            # Check if collection has data
            existing_count = self.vector_store.collection.count()
            print(f"📊 Found existing ChromaDB with {existing_count} embedded chunks")
            
            if existing_count > 0:
                print("✅ Using existing embeddings - system ready!")
                
                # Try to get some sample chunks for TF-IDF (optional)
                try:
                    # Get a sample of documents for TF-IDF backup
                    sample_results = self.vector_store.collection.query(
                        query_texts=["قانون"],
                        n_results=min(200, existing_count)
                    )
                    
                    if sample_results and 'documents' in sample_results:
                        # Build minimal TF-IDF index from available chunks
                        from sklearn.feature_extraction.text import TfidfVectorizer
                        
                        self.vector_store.tfidf_vectorizer = TfidfVectorizer(
                            max_features=5000,
                            stop_words=None,
                            ngram_range=(1, 2)
                        )
                        
                        documents = sample_results['documents'][0]  # Flatten if nested
                        self.vector_store.doc_chunks = documents
                        self.vector_store.tfidf_matrix = self.vector_store.tfidf_vectorizer.fit_transform(documents)
                        
                        print(f"✅ Built TF-IDF index with {len(documents)} chunks")
                    
                except Exception as e:
                    print(f"⚠️  Could not build TF-IDF index: {e}")
                    print("   System will use semantic search only")
                
                return True
            else:
                print("❌ ChromaDB exists but contains no embeddings")
                return False
                
        except Exception as e:
            print(f"❌ Error loading ChromaDB: {e}")
            return False
    
    def load_documents(self, force_rebuild: bool = False) -> int:
        """Override to use embeddings only"""
        success = self.load_embeddings_only()
        return 1 if success else 0

def initialize_deployment_system():
    """Initialize system for deployment"""
    print("🚀 Initializing Deployment Legal RAG System...")
    
    # Check if ChromaDB exists
    chroma_path = Path("./chroma_db")
    if not chroma_path.exists():
        print("❌ ChromaDB not found. Please ensure embeddings are available.")
        return None
    
    system = DeploymentRAGSystem()
    
    # Load embeddings
    num_loaded = system.load_documents()
    
    if num_loaded > 0:
        print("✅ Deployment system ready!")
        return system
    else:
        print("❌ Failed to load embeddings")
        return None

if __name__ == "__main__":
    system = initialize_deployment_system()
    
    if system:
        print("\n🎉 Deployment RAG System ready!")
        print("   - Using existing ChromaDB embeddings")
        print("   - No original documents required")
        print("   - Ready for production deployment")
        
        # Test query
        test_result = system.query("ما هي شروط تأسيس الشركات؟")
        print(f"\n📝 Test query confidence: {test_result.confidence}")
    else:
        print("❌ System initialization failed") 