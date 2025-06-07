#!/usr/bin/env python3
"""
ChromaDB Schema Fix Script for Render Deployment

This script fixes the ChromaDB schema mismatch error by recreating the collection
with the correct schema compatible with the Render environment.
"""

import os
import sys
import shutil
from pathlib import Path
import chromadb
from chromadb.config import Settings

def backup_existing_db():
    """Create backup of existing ChromaDB"""
    chroma_path = os.getenv('CHROMA_DB_PATH', './chroma_db')
    backup_path = f"{chroma_path}_backup_schema_fix"
    
    if os.path.exists(chroma_path):
        if os.path.exists(backup_path):
            print(f"Removing old backup: {backup_path}")
            shutil.rmtree(backup_path)
        
        print(f"Creating backup: {chroma_path} -> {backup_path}")
        shutil.copytree(chroma_path, backup_path)
        return backup_path
    else:
        print(f"No existing ChromaDB found at: {chroma_path}")
        return None

def reset_chromadb():
    """Reset ChromaDB to fix schema issues"""
    chroma_path = os.getenv('CHROMA_DB_PATH', './chroma_db')
    
    print(f"🔄 Resetting ChromaDB at: {chroma_path}")
    
    # Create backup first
    backup_path = backup_existing_db()
    
    try:
        # Initialize new ChromaDB client
        client = chromadb.PersistentClient(
            path=chroma_path,
            settings=Settings(allow_reset=True)
        )
        
        # Reset the database to fix schema issues
        print("🗑️  Resetting ChromaDB database...")
        client.reset()
        
        # Create a fresh collection
        collection = client.create_collection(
            name="jordanian_legal_docs",
            metadata={"description": "Jordanian Legal Documents - Schema Fixed"}
        )
        
        print("✅ ChromaDB reset successfully!")
        print(f"📊 New collection created: {collection.name}")
        
        # Verify the fix
        collections = client.list_collections()
        print(f"📋 Available collections: {[c.name for c in collections]}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error resetting ChromaDB: {e}")
        
        # Restore backup if reset failed
        if backup_path and os.path.exists(backup_path):
            print(f"🔄 Restoring backup from: {backup_path}")
            if os.path.exists(chroma_path):
                shutil.rmtree(chroma_path)
            shutil.copytree(backup_path, chroma_path)
        
        return False

def test_chromadb_connection():
    """Test ChromaDB connection after fix"""
    chroma_path = os.getenv('CHROMA_DB_PATH', './chroma_db')
    
    try:
        print(f"🧪 Testing connection to: {chroma_path}")
        
        client = chromadb.PersistentClient(
            path=chroma_path,
            settings=Settings(allow_reset=True)
        )
        
        # List collections to test schema
        collections = client.list_collections()
        print(f"✅ Connection successful! Collections: {[c.name for c in collections]}")
        
        # Test getting a collection
        if collections:
            collection = client.get_collection(collections[0].name)
            count = collection.count()
            print(f"📊 Collection '{collections[0].name}' has {count} items")
        
        return True
        
    except Exception as e:
        print(f"❌ Connection test failed: {e}")
        return False

def main():
    """Main execution function"""
    print("🔧 ChromaDB Schema Fix Script")
    print("=" * 50)
    
    # Check environment
    chroma_path = os.getenv('CHROMA_DB_PATH', './chroma_db')
    print(f"📍 ChromaDB Path: {chroma_path}")
    
    # Test current connection
    print("\n1. Testing current ChromaDB connection...")
    if test_chromadb_connection():
        print("✅ ChromaDB is working! No fix needed.")
        return True
    
    # Attempt to fix
    print("\n2. Fixing ChromaDB schema...")
    if reset_chromadb():
        print("\n3. Testing fixed ChromaDB...")
        if test_chromadb_connection():
            print("\n✅ ChromaDB fix completed successfully!")
            print("🚀 You can now redeploy your application.")
            return True
        else:
            print("\n❌ Fix verification failed.")
            return False
    else:
        print("\n❌ ChromaDB fix failed.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 