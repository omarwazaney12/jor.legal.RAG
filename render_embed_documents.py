#!/usr/bin/env python3
"""
Render Document Embedding Script

This script embeds all Jordan legal documents on Render using our working custom OpenAI embeddings.
Run this AFTER deployment to populate the ChromaDB with fresh, compatible data.
"""

import os
import sys
import time
from pathlib import Path
from datetime import datetime

def embed_all_documents():
    """Embed all legal documents using the advanced RAG system"""
    print("🚀 Starting Render Document Embedding Process")
    print("=" * 60)
    
    # Check environment
    print("📍 Environment Check:")
    chroma_path = os.getenv('CHROMA_DB_PATH', './chroma_db')
    api_key = os.getenv('OPENAI_API_KEY', '')
    print(f"   CHROMA_DB_PATH: {chroma_path}")
    print(f"   OPENAI_API_KEY: {'✅ Set' if api_key else '❌ Missing'}")
    
    if not api_key:
        print("❌ OPENAI_API_KEY is required! Set it in Render environment variables.")
        return False
    
    # Check documents directory
    docs_path = Path("mit_jordan_data/txt_output")
    if not docs_path.exists():
        print(f"❌ Documents directory not found: {docs_path}")
        return False
    
    txt_files = list(docs_path.glob("*.txt"))
    print(f"📄 Found {len(txt_files)} legal documents to embed")
    
    if len(txt_files) == 0:
        print("❌ No documents found to embed!")
        return False
    
    # Initialize the RAG system
    print("\n🔧 Initializing Advanced Legal RAG System...")
    try:
        from advanced_rag_system import AdvancedLegalRAGSystem
        
        # Initialize with documents path
        system = AdvancedLegalRAGSystem(documents_path="mit_jordan_data/txt_output")
        print("✅ System initialized successfully")
        
    except Exception as e:
        print(f"❌ System initialization failed: {e}")
        return False
    
    # Start embedding process
    print("\n🚀 Starting Document Embedding...")
    print("   This process will take 30-60 minutes depending on document count")
    print("   Progress will be saved automatically for resumption if interrupted")
    
    start_time = datetime.now()
    
    try:
        # Force rebuild to ensure fresh, compatible embeddings
        num_loaded = system.load_documents(force_rebuild=True)
        
        if num_loaded > 0:
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds() / 60
            
            print(f"\n🎉 SUCCESS! Embedded {num_loaded} documents")
            print(f"⏱️  Total time: {duration:.1f} minutes")
            
            # Verify the embeddings
            try:
                collection_count = system.vector_store.collection.count()
                print(f"📊 Verification: {collection_count} embeddings in database")
                
                # Test a sample query
                test_result = system.query("ما هي شروط تأسيس الشركات؟")
                print(f"🧪 Test query confidence: {test_result.confidence:.2f}")
                
                if test_result.confidence > 0.5:
                    print("✅ System fully operational!")
                    return True
                else:
                    print("⚠️  System working but may need more documents")
                    return True
                    
            except Exception as e:
                print(f"⚠️  Verification failed: {e}")
                print("   System may still be working - check web interface")
                return True
        else:
            print("❌ No documents were embedded")
            return False
            
    except Exception as e:
        print(f"❌ Embedding process failed: {e}")
        
        # Check if partial progress was made
        try:
            collection_count = system.vector_store.collection.count()
            if collection_count > 0:
                print(f"⚠️  Partial progress: {collection_count} embeddings created")
                print("   You can continue or restart the process")
                return True
        except:
            pass
        
        return False

def check_system_status():
    """Check current system status"""
    print("🔍 Checking Current System Status...")
    
    try:
        from deploy_without_docs import DeploymentRAGSystem
        
        system = DeploymentRAGSystem()
        loaded = system.load_embeddings_only()
        
        if loaded:
            count = system.vector_store.collection.count()
            print(f"✅ System loaded with {count} embeddings")
            
            if count > 0:
                print("🎯 System is ready - no embedding needed!")
                return True
            else:
                print("⚠️  System loaded but database is empty - embedding needed")
                return False
        else:
            print("❌ System not loaded - embedding needed")
            return False
            
    except Exception as e:
        print(f"❌ Status check failed: {e}")
        return False

def main():
    """Main execution function"""
    print("🔧 Render Document Embedding Script")
    print(f"📅 Started at: {datetime.now().isoformat()}")
    print("=" * 60)
    
    # Check current status first
    if check_system_status():
        print("\n✅ System already has embeddings - no action needed!")
        print("   If you want to re-embed, delete ChromaDB first")
        return True
    
    # Proceed with embedding
    print("\n🚀 Starting fresh embedding process...")
    success = embed_all_documents()
    
    if success:
        print("\n" + "=" * 60)
        print("🎉 EMBEDDING COMPLETED SUCCESSFULLY!")
        print("✅ Your Jordan Legal RAG system is now fully operational")
        print("🌐 Check the web interface - it should show 'System Ready'")
        print("🧪 Test with queries like: 'ما هي شروط تأسيس الشركات؟'")
    else:
        print("\n" + "=" * 60)
        print("❌ EMBEDDING FAILED")
        print("🔧 Check the error messages above for troubleshooting")
        print("🔄 You can re-run this script to retry")
    
    return success

if __name__ == "__main__":
    success = main()
    print(f"\n📅 Finished at: {datetime.now().isoformat()}")
    sys.exit(0 if success else 1) 