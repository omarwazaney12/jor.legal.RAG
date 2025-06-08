#!/usr/bin/env python3
"""
Render-specific ChromaDB initialization script
Configured for ChromaDB 0.4.17 compatibility with Render's schema
"""

import os
import sys
import chromadb
from pathlib import Path

def initialize_chromadb_for_render():
    """Initialize ChromaDB with Render-compatible 0.4.17 settings"""
    
    # Use environment variable or default path
    chroma_path = os.getenv('CHROMA_DB_PATH', './chroma_db')
    
    print(f"🔧 Initializing ChromaDB 0.4.17 for Render deployment...")
    print(f"📂 Database path: {chroma_path}")
    
    # Ensure directory exists
    Path(chroma_path).mkdir(parents=True, exist_ok=True)
    
    # Check if we need to clean up old schema
    db_files = list(Path(chroma_path).glob('*'))
    if db_files:
        print(f"⚠️  Found existing database files: {len(db_files)} items")
        print("🧹 Cleaning up to prevent schema conflicts...")
        
        # Remove old files to prevent schema mismatch
        for file_path in db_files:
            try:
                if file_path.is_file():
                    file_path.unlink()
                elif file_path.is_dir():
                    import shutil
                    shutil.rmtree(file_path)
                print(f"   ✅ Removed: {file_path.name}")
            except Exception as e:
                print(f"   ⚠️  Could not remove {file_path.name}: {e}")
    
    # Initialize with 0.4.17 modern syntax
    try:
        client = chromadb.PersistentClient(path=chroma_path)
        
        print("✅ ChromaDB 0.4.17 client initialized successfully")
        
        # Create the collection if it doesn't exist
        collection_name = "jordanian_legal_docs"
        try:
            collections = client.list_collections()
            existing_names = [c.name for c in collections]
            
            if collection_name in existing_names:
                print(f"📋 Collection '{collection_name}' already exists")
                collection = client.get_collection(collection_name)
                count = collection.count()
                print(f"📊 Collection has {count} documents")
            else:
                print(f"🆕 Creating new collection: {collection_name}")
                collection = client.create_collection(
                    name=collection_name,
                    metadata={
                        "description": "Jordanian Legal Documents - Render 0.4.17 Compatible",
                        "version": "0.4.17"
                    }
                )
                print(f"✅ Collection '{collection_name}' created successfully")
                
        except Exception as e:
            print(f"❌ Collection operations failed: {e}")
            return False
            
        print("🎉 ChromaDB 0.4.17 initialization completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ ChromaDB initialization failed: {e}")
        return False

def test_chromadb_connection():
    """Test the ChromaDB connection"""
    try:
        chroma_path = os.getenv('CHROMA_DB_PATH', './chroma_db')
        
        client = chromadb.PersistentClient(path=chroma_path)
        collections = client.list_collections()
        
        print(f"✅ Connection test passed. Found {len(collections)} collections:")
        for collection in collections:
            try:
                count = client.get_collection(collection.name).count()
                print(f"   - {collection.name}: {count} documents")
            except Exception as e:
                print(f"   - {collection.name}: Error accessing ({e})")
                
        return True
        
    except Exception as e:
        print(f"❌ Connection test failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Render ChromaDB 0.4.17 Initialization Script")
    print("=" * 50)
    
    # Initialize ChromaDB
    if initialize_chromadb_for_render():
        print("\n🧪 Testing connection...")
        if test_chromadb_connection():
            print("\n🎉 All tests passed! ChromaDB 0.4.17 is ready for Render deployment.")
            sys.exit(0)
        else:
            print("\n❌ Connection test failed.")
            sys.exit(1)
    else:
        print("\n❌ Initialization failed.")
        sys.exit(1) 