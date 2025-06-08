#!/usr/bin/env python3
"""
Render Deployment Script - Uses Pre-built ChromaDB Database
This version loads a pre-built ChromaDB database instead of creating embeddings on Render
"""

import os
import sys
from pathlib import Path
from typing import Optional
import chromadb

class DeploymentRAGSystem:
    """Deployment version that loads pre-built ChromaDB database"""
    
    def __init__(self):
        self.chroma_client = None
        self.collection = None
        self.is_ready = False
        
    def load_prebuilt_database(self) -> bool:
        """Load the pre-built ChromaDB database created locally"""
        try:
            # Use Render's expected database path or local path for testing
            chroma_path = os.getenv('CHROMA_DB_PATH', './chroma_db_render')
            
            print(f"📂 Loading pre-built database from: {chroma_path}")
            
            # Check if database exists
            if not Path(chroma_path).exists():
                print(f"❌ Database not found at {chroma_path}")
                print("💡 Make sure you've run build_embeddings_locally.py first")
                return False
            
            # Initialize ChromaDB with modern 0.4.17 syntax
            self.chroma_client = chromadb.PersistentClient(path=chroma_path)
            
            # Get the collection
            collections = self.chroma_client.list_collections()
            print(f"✅ Found {len(collections)} collections")
            
            if not collections:
                print("❌ No collections found in database")
                return False
            
            # Get the main collection
            collection_name = "jordanian_legal_docs"
            try:
                self.collection = self.chroma_client.get_collection(collection_name)
                doc_count = self.collection.count()
                print(f"✅ Loaded collection '{collection_name}' with {doc_count} documents")
                
                if doc_count == 0:
                    print("⚠️  Collection is empty!")
                    return False
                
                self.is_ready = True
                return True
                
            except Exception as e:
                print(f"❌ Could not access collection '{collection_name}': {e}")
                return False
                
        except Exception as e:
            print(f"❌ Failed to load pre-built database: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def query(self, query_text: str, n_results: int = 5):
        """Query the pre-built database"""
        if not self.is_ready or not self.collection:
            return {
                'confidence': 0.0,
                'answer': 'النظام غير متاح حالياً. يرجى المحاولة لاحقاً.',
                'sources': [],
                'error': 'Database not ready'
            }
        
        try:
            # Perform semantic search
            results = self.collection.query(
                query_texts=[query_text],
                n_results=n_results,
                include=['documents', 'metadatas', 'distances']
            )
            
            if not results['documents'] or not results['documents'][0]:
                return {
                    'confidence': 0.0,
                    'answer': 'لم يتم العثور على معلومات ذات صلة بالاستعلام.',
                    'sources': []
                }
            
            # Calculate confidence from distances
            distances = results['distances'][0] if results['distances'] else [1.0] * len(results['documents'][0])
            avg_distance = sum(distances) / len(distances)
            confidence = max(0.0, 1.0 - avg_distance)  # Convert distance to confidence
            
            # Format sources
            sources = []
            documents = results['documents'][0]
            metadatas = results['metadatas'][0] if results['metadatas'] else [{}] * len(documents)
            
            for i, (doc, metadata) in enumerate(zip(documents, metadatas)):
                sources.append({
                    'content': doc,
                    'metadata': metadata,
                    'relevance_score': 1.0 - distances[i] if i < len(distances) else 0.5
                })
            
            # Create a simple answer from the most relevant documents
            top_docs = documents[:3]  # Use top 3 most relevant
            answer = f"بناءً على الوثائق القانونية المتاحة:\n\n{' '.join(top_docs[:2])}"
            
            return {
                'confidence': confidence,
                'answer': answer,
                'sources': sources
            }
            
        except Exception as e:
            print(f"❌ Query failed: {e}")
            return {
                'confidence': 0.0,
                'answer': f'حدث خطأ أثناء البحث: {str(e)}',
                'sources': [],
                'error': str(e)
            }
    
    def get_system_stats(self):
        """Get system statistics"""
        if not self.is_ready:
            return {
                'status': 'not_ready',
                'database_path': os.getenv('CHROMA_DB_PATH', './chroma_db_render'),
                'collections': 0,
                'total_documents': 0
            }
        
        try:
            collections = self.chroma_client.list_collections()
            total_docs = sum(self.chroma_client.get_collection(c.name).count() for c in collections)
            
            return {
                'status': 'ready',
                'database_path': os.getenv('CHROMA_DB_PATH', './chroma_db_render'),
                'collections': len(collections),
                'total_documents': total_docs,
                'backend': 'duckdb+parquet'
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'database_path': os.getenv('CHROMA_DB_PATH', './chroma_db_render')
            }

def test_deployment_system():
    """Test the deployment system"""
    print("🧪 Testing Deployment RAG System")
    print("=" * 40)
    
    system = DeploymentRAGSystem()
    
    # Load database
    if system.load_prebuilt_database():
        print("✅ Database loaded successfully")
        
        # Test query
        test_query = "ما هي شروط تأسيس الشركات؟"
        print(f"\n🔍 Testing query: {test_query}")
        
        result = system.query(test_query)
        print(f"✅ Query result:")
        print(f"   Confidence: {result['confidence']:.2f}")
        print(f"   Sources: {len(result['sources'])}")
        print(f"   Answer preview: {result['answer'][:100]}...")
        
        # Show stats
        stats = system.get_system_stats()
        print(f"\n📊 System Stats:")
        print(f"   Status: {stats['status']}")
        print(f"   Collections: {stats['collections']}")
        print(f"   Total documents: {stats['total_documents']}")
        print(f"   Backend: {stats.get('backend', 'unknown')}")
        
        return True
    else:
        print("❌ Failed to load database")
        return False

if __name__ == "__main__":
    # Test the system
    if test_deployment_system():
        print("\n🎉 Deployment system ready!")
    else:
        print("\n❌ Deployment system failed!")
        sys.exit(1) 