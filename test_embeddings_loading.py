#!/usr/bin/env python3
"""
Test script to verify embeddings loading and basic functionality
"""

import os
import sys
from pathlib import Path

def test_embeddings_file():
    """Test if embeddings file exists and is valid"""
    print("🔍 Testing embeddings file...")
    
    possible_paths = [
        Path("railway_embeddings_data/embeddings_data.json"),
        Path("./railway_embeddings_data/embeddings_data.json"),
        Path("/app/railway_embeddings_data/embeddings_data.json"),
        Path("embeddings_data.json")
    ]
    
    embeddings_file = None
    for path in possible_paths:
        print(f"   Checking: {path}")
        if path.exists():
            embeddings_file = path
            print(f"   ✅ Found: {path}")
            break
        else:
            print(f"   ❌ Not found: {path}")
    
    if not embeddings_file:
        print("❌ No embeddings file found!")
        print(f"📁 Current directory: {os.getcwd()}")
        print("📁 Directory contents:")
        for item in os.listdir("."):
            print(f"   - {item}")
        return False
    
    # Test loading the JSON
    try:
        import json
        with open(embeddings_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"✅ Embeddings file loaded successfully:")
        print(f"   - Documents: {len(data['documents'])}")
        print(f"   - Model: {data.get('model', 'unknown')}")
        print(f"   - Dimension: {data.get('embedding_dim', 'unknown')}")
        return True
        
    except Exception as e:
        print(f"❌ Error loading embeddings: {e}")
        return False

def test_rag_system():
    """Test the RAG system initialization"""
    print("\n🤖 Testing RAG system...")
    
    try:
        from advanced_rag_system import AdvancedLegalRAGSystem
        
        print("   Creating system...")
        system = AdvancedLegalRAGSystem()
        
        print("   Loading documents...")
        num_docs = system.load_documents()
        
        if num_docs > 0:
            print(f"   ✅ System loaded {num_docs} documents")
            
            # Test a simple query
            print("   Testing query...")
            result = system.query("ما هو قانون الشركات؟")
            
            if result and result.answer:
                print(f"   ✅ Query successful: {result.answer[:100]}...")
                print(f"   ✅ Confidence: {result.confidence}")
                print(f"   ✅ Sources: {len(result.sources)} documents")
                return True
            else:
                print("   ❌ Query failed - no answer")
                return False
        else:
            print("   ❌ No documents loaded")
            return False
            
    except Exception as e:
        print(f"   ❌ RAG system error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_openai_key():
    """Test OpenAI API key"""
    print("\n🔑 Testing OpenAI API key...")
    
    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        print("   ❌ OPENAI_API_KEY not set in environment")
        return False
    
    if api_key.startswith('sk-'):
        print(f"   ✅ API key format looks correct: {api_key[:10]}...")
        return True
    else:
        print(f"   ⚠️  API key format unusual: {api_key[:10]}...")
        return True

def main():
    """Run all tests"""
    print("🧪 Running system tests...\n")
    
    results = []
    
    # Test 1: Embeddings file
    results.append(("Embeddings File", test_embeddings_file()))
    
    # Test 2: OpenAI key
    results.append(("OpenAI API Key", test_openai_key()))
    
    # Test 3: RAG system
    results.append(("RAG System", test_rag_system()))
    
    # Summary
    print("\n" + "="*50)
    print("📊 TEST RESULTS:")
    print("="*50)
    
    all_passed = True
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name:<20} {status}")
        if not passed:
            all_passed = False
    
    print("="*50)
    if all_passed:
        print("🎉 All tests passed! System should work correctly.")
    else:
        print("⚠️  Some tests failed. Check the issues above.")
    print("="*50)

if __name__ == "__main__":
    main() 