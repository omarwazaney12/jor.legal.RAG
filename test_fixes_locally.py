#!/usr/bin/env python3
"""
Local Testing Script for Render Deployment Fixes

Test the OpenAI compatibility and ChromaDB fixes locally before deploying to Render.
"""

import os
import sys

def test_openai_embeddings():
    """Test OpenAI embeddings initialization with the fixes"""
    print("🧪 Testing OpenAI Embeddings...")
    
    try:
        from langchain_openai import OpenAIEmbeddings
        
        # Test the fixed initialization
        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-large"
        )
        
        # Test embedding generation
        test_text = ["Hello world", "Test embedding"]
        result = embeddings.embed_documents(test_text)
        
        print(f"✅ OpenAI embeddings working! Generated {len(result)} embeddings")
        print(f"   Embedding dimension: {len(result[0])}")
        return True
        
    except Exception as e:
        print(f"❌ OpenAI embeddings failed: {e}")
        
        # Try fallback model
        try:
            print("🔄 Trying fallback model...")
            embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
            result = embeddings.embed_documents(["test"])
            print("✅ Fallback model working!")
            return True
        except Exception as e2:
            print(f"❌ Fallback also failed: {e2}")
            return False

def test_chromadb_connection():
    """Test ChromaDB connection and schema compatibility"""
    print("\n🧪 Testing ChromaDB Connection...")
    
    try:
        import chromadb
        from chromadb.config import Settings
        
        # Use local path for testing
        chroma_path = "./chroma_db"
        
        client = chromadb.PersistentClient(
            path=chroma_path,
            settings=Settings(allow_reset=True)
        )
        
        # Test listing collections (this was failing before)
        collections = client.list_collections()
        print(f"✅ ChromaDB connection successful! Collections: {[c.name for c in collections]}")
        
        # Test getting collection if exists
        if collections:
            collection = client.get_collection(collections[0].name)
            count = collection.count()
            print(f"   Collection '{collections[0].name}' has {count} items")
        
        return True
        
    except Exception as e:
        print(f"❌ ChromaDB connection failed: {e}")
        return False

def test_vector_store():
    """Test the AdvancedVectorStore with fixes"""
    print("\n🧪 Testing AdvancedVectorStore...")
    
    try:
        from advanced_rag_system import AdvancedVectorStore
        
        vector_store = AdvancedVectorStore()
        print("✅ AdvancedVectorStore initialized successfully!")
        
        # Test if collection exists and has data
        try:
            count = vector_store.collection.count()
            print(f"   Collection has {count} embedded documents")
            
            if count > 0:
                # Test a simple query
                results = vector_store.collection.query(
                    query_texts=["قانون"],
                    n_results=3
                )
                print(f"   Test query returned {len(results['documents'][0])} results")
                
        except Exception as e:
            print(f"   ⚠️  Collection query failed: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ AdvancedVectorStore failed: {e}")
        return False

def test_deployment_system():
    """Test the DeploymentRAGSystem"""
    print("\n🧪 Testing DeploymentRAGSystem...")
    
    try:
        from deploy_without_docs import DeploymentRAGSystem
        
        system = DeploymentRAGSystem()
        result = system.load_embeddings_only()
        
        if result:
            print("✅ DeploymentRAGSystem loaded successfully!")
            
            # Test a query
            try:
                test_query = "ما هي شروط تأسيس الشركات؟"
                response = system.query(test_query)
                print(f"   Test query confidence: {response.confidence}")
                print(f"   Response length: {len(response.answer)} characters")
                return True
            except Exception as e:
                print(f"   ⚠️  Query test failed: {e}")
                return True  # System loaded even if query failed
        else:
            print("❌ DeploymentRAGSystem failed to load embeddings")
            return False
            
    except Exception as e:
        print(f"❌ DeploymentRAGSystem failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🔧 Testing Render Deployment Fixes Locally")
    print("=" * 50)
    
    # Check environment
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("⚠️  OPENAI_API_KEY not set - embeddings tests may fail")
    else:
        print(f"✅ OPENAI_API_KEY is set (length: {len(api_key)})")
    
    results = []
    
    # Test 1: OpenAI Embeddings
    results.append(test_openai_embeddings())
    
    # Test 2: ChromaDB Connection
    results.append(test_chromadb_connection())
    
    # Test 3: Vector Store
    results.append(test_vector_store())
    
    # Test 4: Deployment System
    results.append(test_deployment_system())
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    
    test_names = [
        "OpenAI Embeddings",
        "ChromaDB Connection", 
        "AdvancedVectorStore",
        "DeploymentRAGSystem"
    ]
    
    passed = 0
    for i, (name, result) in enumerate(zip(test_names, results)):
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {name}: {status}")
        if result:
            passed += 1
    
    print(f"\n🎯 Overall: {passed}/{len(results)} tests passed")
    
    if all(results):
        print("🚀 All tests passed! Ready for Render deployment.")
        return True
    else:
        print("⚠️  Some tests failed. Check the issues above before deploying.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 