#!/usr/bin/env python3
"""
Local Embedding Builder for Render Deployment
Creates ChromaDB database locally with proper DuckDB schema that works on Render
"""

import os
import sys
import time
from pathlib import Path
from advanced_rag_system import AdvancedLegalRAGSystem

def build_embeddings_for_render():
    """Build all embeddings locally with Render-compatible schema"""
    
    print("🚀 Building ChromaDB Embeddings for Render Deployment")
    print("=" * 60)
    
    # Set the database path to what Render expects
    render_db_path = "./chroma_db_render"
    os.environ['CHROMA_DB_PATH'] = render_db_path
    
    print(f"📂 Building database at: {render_db_path}")
    print(f"🔧 Using DuckDB backend (Render-compatible)")
    
    # Clean up any existing database
    if Path(render_db_path).exists():
        print("🧹 Cleaning up existing database...")
        import shutil
        shutil.rmtree(render_db_path)
    
    # Initialize the RAG system (this will create the ChromaDB with DuckDB backend)
    print("⚡ Initializing RAG system with Render-compatible configuration...")
    start_time = time.time()
    
    try:
        system = AdvancedLegalRAGSystem()
        print("✅ RAG system initialized successfully")
        
        # Force rebuild all embeddings
        print("\n📚 Loading and embedding all documents...")
        print("⏳ This may take several minutes...")
        
        doc_count = system.load_documents(force_rebuild=True)
        
        if doc_count > 0:
            elapsed = time.time() - start_time
            print(f"\n🎉 Successfully processed {doc_count} documents!")
            print(f"⏱️  Total time: {elapsed:.1f} seconds")
            
            # Test the database
            print("\n🧪 Testing the built database...")
            test_query = "ما هي شروط تأسيس الشركات؟"
            result = system.query(test_query)
            
            print(f"✅ Test query successful!")
            print(f"   Confidence: {result.confidence:.2f}")
            print(f"   Sources: {len(result.sources)}")
            print(f"   Answer preview: {result.answer[:100]}...")
            
            # Get database stats
            stats = system.get_system_stats()
            print(f"\n📊 Database Statistics:")
            print(f"   Total documents: {stats.get('total_documents', 'N/A')}")
            print(f"   Total chunks: {stats.get('total_chunks', 'N/A')}")
            print(f"   Vector store status: {stats.get('vector_store_status', 'N/A')}")
            
            # Show database files
            db_files = list(Path(render_db_path).glob("**/*"))
            total_size = sum(f.stat().st_size for f in db_files if f.is_file()) / (1024 * 1024)  # MB
            print(f"   Database size: {total_size:.1f} MB")
            print(f"   Database files: {len([f for f in db_files if f.is_file()])}")
            
            print(f"\n🎯 Ready for Render deployment!")
            print(f"📁 Database location: {render_db_path}")
            print(f"💡 Next steps:")
            print(f"   1. Commit and push the {render_db_path} folder")
            print(f"   2. Deploy to Render")
            print(f"   3. Render will load the pre-built database")
            
            return True
            
        else:
            print("❌ No documents were processed!")
            return False
            
    except Exception as e:
        print(f"❌ Error building embeddings: {e}")
        import traceback
        traceback.print_exc()
        return False

def verify_render_compatibility():
    """Verify the database is compatible with Render's expected configuration"""
    
    print("\n🔍 Verifying Render compatibility...")
    
    render_db_path = "./chroma_db_render"
    
    if not Path(render_db_path).exists():
        print("❌ Database not found. Run embedding build first.")
        return False
    
    try:
        # Test with the exact configuration Render will use
        import chromadb
        from chromadb.config import Settings
        
        client = chromadb.PersistentClient(
            path=render_db_path,
            settings=Settings(
                chroma_db_impl="duckdb+parquet",
                allow_reset=True
            )
        )
        
        collections = client.list_collections()
        print(f"✅ Found {len(collections)} collections")
        
        for collection in collections:
            count = client.get_collection(collection.name).count()
            print(f"   - {collection.name}: {count} documents")
            
        print("✅ Database is fully compatible with Render!")
        return True
        
    except Exception as e:
        print(f"❌ Compatibility check failed: {e}")
        return False

if __name__ == "__main__":
    # Build embeddings
    if build_embeddings_for_render():
        # Verify compatibility
        if verify_render_compatibility():
            print("\n🚀 SUCCESS: Database ready for Render deployment!")
            sys.exit(0)
        else:
            print("\n❌ FAILED: Database compatibility issues")
            sys.exit(1)
    else:
        print("\n❌ FAILED: Could not build embeddings")
        sys.exit(1) 