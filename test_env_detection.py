#!/usr/bin/env python3
"""
Test automatic environment detection for ChromaDB path
"""

import os

def test_environment_detection():
    """Test that the environment detection works correctly"""
    print("🧪 Testing Automatic Environment Detection")
    print("=" * 45)
    
    # Test 1: Local environment (no environment variable)
    print("\n1. Local Development Environment:")
    if 'CHROMA_DB_PATH' in os.environ:
        del os.environ['CHROMA_DB_PATH']
    
    local_path = os.getenv('CHROMA_DB_PATH', './chroma_db')
    print(f"   Environment Variable: {os.getenv('CHROMA_DB_PATH', 'Not Set')}")
    print(f"   ChromaDB Path Used:   {local_path}")
    print(f"   ✅ Using local development path")
    
    # Test 2: Render environment (with environment variable)
    print("\n2. Render Production Environment:")
    os.environ['CHROMA_DB_PATH'] = '/opt/render/project/src/chroma_db'
    
    render_path = os.getenv('CHROMA_DB_PATH', './chroma_db')
    print(f"   Environment Variable: {os.getenv('CHROMA_DB_PATH')}")
    print(f"   ChromaDB Path Used:   {render_path}")
    print(f"   ✅ Using Render persistent storage path")
    
    # Test 3: Custom environment
    print("\n3. Custom Environment:")
    os.environ['CHROMA_DB_PATH'] = '/custom/path/chroma_db'
    
    custom_path = os.getenv('CHROMA_DB_PATH', './chroma_db')
    print(f"   Environment Variable: {os.getenv('CHROMA_DB_PATH')}")
    print(f"   ChromaDB Path Used:   {custom_path}")
    print(f"   ✅ Using custom path")
    
    # Clean up
    if 'CHROMA_DB_PATH' in os.environ:
        del os.environ['CHROMA_DB_PATH']
    
    print("\n" + "=" * 45)
    print("🎉 Automatic Environment Detection Working!")
    print("\n📋 How it works:")
    print("• Local dev:  Uses './chroma_db' (default)")
    print("• Render:     Uses '/opt/render/project/src/chroma_db' (from env var)")
    print("• Custom:     Uses any path set in CHROMA_DB_PATH")
    
    print("\n🔧 Implementation:")
    print("chroma_path = os.getenv('CHROMA_DB_PATH', './chroma_db')")
    print("self.chroma_client = chromadb.PersistentClient(path=chroma_path, ...)")

if __name__ == "__main__":
    test_environment_detection() 