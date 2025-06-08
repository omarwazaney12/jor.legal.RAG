#!/usr/bin/env python3
"""
Build embeddings for Render deployment using server mode
This avoids local SQLite FTS5 issues by using ChromaDB server
"""

import os
import sys
import time
import subprocess
import requests
from pathlib import Path

def start_chromadb_server():
    """Start ChromaDB server in the background"""
    
    db_path = "./chroma_db_render"
    
    print("🚀 Starting ChromaDB server...")
    
    # Create database directory
    Path(db_path).mkdir(exist_ok=True)
    
    # Start server
    cmd = f"chroma run --path {db_path} --port 8000"
    
    print(f"📁 Database path: {db_path}")
    print(f"🌐 Server command: {cmd}")
    
    # Start server as subprocess
    process = subprocess.Popen(
        cmd.split(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    # Wait for server to start
    print("⏳ Waiting for server to start...")
    for i in range(30):  # Wait up to 30 seconds
        try:
            response = requests.get("http://localhost:8000/api/v1/heartbeat", timeout=1)
            if response.status_code == 200:
                print("✅ ChromaDB server is running!")
                return process
        except:
            pass
        time.sleep(1)
        print(f"   ... waiting ({i+1}/30)")
    
    print("❌ Server failed to start")
    process.terminate()
    return None

def build_embeddings_with_server():
    """Build embeddings using ChromaDB server"""
    
    print("🔧 Building embeddings via ChromaDB server...")
    
    # Import here to avoid import issues
    import chromadb
    from advanced_rag_system import AdvancedLegalRAGSystem
    
    # Override environment to use server mode
    os.environ['CHROMA_DB_PATH'] = './chroma_db_render'
    
    try:
        # Initialize system - it should connect to server
        system = AdvancedLegalRAGSystem()
        
        # Build embeddings
        print("📚 Processing documents...")
        doc_count = system.load_documents(force_rebuild=True)
        
        if doc_count > 0:
            print(f"✅ Successfully processed {doc_count} documents!")
            
            # Test the system
            result = system.query("ما هي شروط تأسيس الشركات؟")
            print(f"🧪 Test query confidence: {result.confidence:.2f}")
            
            return True
        else:
            print("❌ No documents processed")
            return False
            
    except Exception as e:
        print(f"❌ Error building embeddings: {e}")
        import traceback
        traceback.print_exc()
        return False

def stop_server(process):
    """Stop ChromaDB server"""
    if process:
        print("🛑 Stopping ChromaDB server...")
        process.terminate()
        process.wait()
        print("✅ Server stopped")

def main():
    """Main function"""
    
    print("🚀 Building ChromaDB for Render Deployment (Server Mode)")
    print("=" * 60)
    
    # Start server
    server_process = start_chromadb_server()
    
    if not server_process:
        print("❌ Could not start ChromaDB server")
        return False
    
    try:
        # Build embeddings
        success = build_embeddings_with_server()
        
        if success:
            print("\n🎉 SUCCESS: Embeddings built successfully!")
            print("📁 Database location: ./chroma_db_render/")
            print("💡 Next steps:")
            print("   1. git add chroma_db_render/")
            print("   2. git commit -m 'Add pre-built ChromaDB for Render'")
            print("   3. git push")
            return True
        else:
            print("\n❌ FAILED: Could not build embeddings")
            return False
            
    finally:
        # Always stop server
        stop_server(server_process)

if __name__ == "__main__":
    if main():
        sys.exit(0)
    else:
        sys.exit(1) 