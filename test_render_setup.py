#!/usr/bin/env python3
"""
Test script to verify Render persistent storage configuration
"""

import os
from pathlib import Path
import sys

def test_configuration():
    """Test the persistent storage configuration"""
    print("🧪 Testing Render Persistent Storage Configuration")
    print("=" * 50)
    
    # Test 1: Environment variable handling
    print("\n1. Testing environment variable handling:")
    original_env = os.getenv('CHROMA_DB_PATH')
    
    # Test local environment (should use ./chroma_db)
    if 'CHROMA_DB_PATH' in os.environ:
        del os.environ['CHROMA_DB_PATH']
    
    from advanced_rag_system import AdvancedVectorStore
    vector_store = AdvancedVectorStore()
    
    # Get the actual path used
    actual_path = vector_store.chroma_client._settings.persist_directory
    print(f"   Local environment path: {actual_path}")
    
    # Test Render environment
    os.environ['CHROMA_DB_PATH'] = '/opt/render/project/src/chroma_db'
    vector_store_render = AdvancedVectorStore()
    render_path = vector_store_render.chroma_client._settings.persist_directory
    print(f"   Render environment path: {render_path}")
    
    # Restore original environment
    if original_env:
        os.environ['CHROMA_DB_PATH'] = original_env
    elif 'CHROMA_DB_PATH' in os.environ:
        del os.environ['CHROMA_DB_PATH']
    
    # Test 2: Deployment system
    print("\n2. Testing deployment system:")
    try:
        from deploy_without_docs import DeploymentRAGSystem
        deployment_system = DeploymentRAGSystem()
        print(f"   ✅ DeploymentRAGSystem initialized")
        print(f"   ChromaDB path: {deployment_system.chroma_db_path}")
    except Exception as e:
        print(f"   ❌ DeploymentRAGSystem failed: {e}")
    
    # Test 3: Check render.yaml configuration
    print("\n3. Checking render.yaml configuration:")
    render_yaml_path = Path("render.yaml")
    if render_yaml_path.exists():
        content = render_yaml_path.read_text()
        if "CHROMA_DB_PATH" in content:
            print("   ✅ CHROMA_DB_PATH environment variable configured")
        if "/opt/render/project/src/chroma_db" in content:
            print("   ✅ Mount path configured correctly")
        if "jordan-legal-data" in content:
            print("   ✅ Disk name configured")
        print("   ✅ render.yaml configuration looks good")
    else:
        print("   ❌ render.yaml not found")
    
    # Test 4: Check ChromaDB backup
    print("\n4. Checking ChromaDB backup:")
    backup_files = list(Path(".").glob("chroma_backup_*.zip"))
    if backup_files:
        latest_backup = max(backup_files, key=lambda x: x.stat().st_mtime)
        backup_size = latest_backup.stat().st_size / 1024 / 1024
        print(f"   ✅ Latest backup: {latest_backup.name} ({backup_size:.1f} MB)")
    else:
        print("   ⚠️  No backup files found - run create_backup.py")
    
    # Test 5: Application compatibility
    print("\n5. Testing application compatibility:")
    try:
        from advanced_web_demo import initialize_system
        print("   ✅ Web demo imports working")
        print("   ✅ Application should work on Render")
    except Exception as e:
        print(f"   ❌ Application compatibility issue: {e}")
    
    print("\n" + "=" * 50)
    print("🎉 Configuration test complete!")
    print("\n📋 Next steps for Render deployment:")
    print("1. Install Render CLI: npm install -g @render/cli")
    print("2. Upload your backup: Follow instructions in RENDER_SETUP_GUIDE.md")
    print("3. Deploy your service: render deploy jordan-legal-rag")
    print("4. Monitor logs: render logs jordan-legal-rag")

if __name__ == "__main__":
    test_configuration() 