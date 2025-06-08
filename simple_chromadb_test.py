#!/usr/bin/env python3
"""
Simple ChromaDB test to find working configuration for 0.4.17
"""

import os
try:
    import chromadb
    print(f"✅ ChromaDB version: {chromadb.__version__}")
except ImportError as e:
    print(f"❌ ChromaDB import failed: {e}")

def test_chromadb_configurations():
    """Test different ChromaDB configurations to find one that works"""
    
    test_path = "./test_chroma_db"
    
    print("🧪 Testing ChromaDB 0.4.17 configurations...")
    
    # Test 1: EphemeralClient (should work but no persistence)
    try:
        print("\n1️⃣ Testing EphemeralClient...")
        client = chromadb.EphemeralClient()
        collection = client.create_collection("test")
        collection.add(
            documents=["Hello world"],
            ids=["1"]
        )
        print("✅ EphemeralClient works!")
        client.delete_collection("test")
    except Exception as e:
        print(f"❌ EphemeralClient failed: {e}")
    
    # Test 2: Basic Client() - let's see what happens
    try:
        print("\n2️⃣ Testing basic Client()...")
        client = chromadb.Client()
        collection = client.create_collection("test2")
        collection.add(
            documents=["Hello world"],
            ids=["1"]
        )
        print("✅ Basic Client() works!")
        client.delete_collection("test2")
    except Exception as e:
        print(f"❌ Basic Client() failed: {e}")
    
    # Test 3: Try to use a different approach for persistence
    try:
        print("\n3️⃣ Testing with custom settings...")
        import chromadb.config
        # Create settings that explicitly use duckdb
        client = chromadb.Client(chromadb.config.Settings(
            is_persistent=True,
            persist_directory=test_path,
            anonymized_telemetry=False
        ))
        collection = client.create_collection("test3")
        collection.add(
            documents=["Hello world"],
            ids=["1"]
        )
        print("✅ Custom settings work!")
        client.delete_collection("test3")
    except Exception as e:
        print(f"❌ Custom settings failed: {e}")

if __name__ == "__main__":
    test_chromadb_configurations() 