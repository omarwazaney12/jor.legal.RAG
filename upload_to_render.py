#!/usr/bin/env python3
"""
Script to help upload ChromaDB to Render
This script can be run on Render to recreate the ChromaDB structure
"""

import os
import shutil
from pathlib import Path

def create_chroma_structure():
    """Create the basic ChromaDB structure on Render"""
    
    print("🚀 Setting up ChromaDB on Render...")
    
    # Check current location
    current_dir = Path.cwd()
    print(f"📍 Current directory: {current_dir}")
    
    # Target directory for ChromaDB
    chroma_target = Path("/opt/render/project/src/chroma_db")
    
    print(f"🎯 Target ChromaDB path: {chroma_target}")
    
    # Create directory structure
    chroma_target.mkdir(parents=True, exist_ok=True)
    print(f"✅ Created directory: {chroma_target}")
    
    # Check if we have local ChromaDB to copy
    local_chroma = Path("chroma_db")
    if local_chroma.exists():
        print(f"📁 Found local ChromaDB, copying...")
        shutil.copytree(local_chroma, chroma_target, dirs_exist_ok=True)
        print(f"✅ Copied ChromaDB to {chroma_target}")
    else:
        print("📝 No local ChromaDB found, creating placeholder")
        
    # List contents
    if chroma_target.exists():
        contents = list(chroma_target.rglob("*"))
        print(f"📊 ChromaDB contents ({len(contents)} items):")
        for item in contents[:10]:  # Show first 10 items
            print(f"   {item.relative_to(chroma_target)}")
        if len(contents) > 10:
            print(f"   ... and {len(contents) - 10} more items")
    
    # Set environment variable
    os.environ['CHROMA_DB_PATH'] = str(chroma_target)
    print(f"🔧 Set CHROMA_DB_PATH={chroma_target}")
    
    return chroma_target

def check_render_environment():
    """Check if we're running on Render"""
    
    print("🔍 Checking environment...")
    
    # Check for Render-specific paths
    render_indicators = [
        "/opt/render",
        "/opt/render/project"
    ]
    
    is_render = any(Path(path).exists() for path in render_indicators)
    
    print(f"🏭 Running on Render: {is_render}")
    
    if is_render:
        print("✅ Render environment detected")
        return True
    else:
        print("💻 Local environment detected")
        return False

if __name__ == "__main__":
    print("🚀 ChromaDB Render Setup Script")
    print("=" * 40)
    
    is_render = check_render_environment()
    
    if is_render:
        chroma_path = create_chroma_structure()
        print(f"\n✅ Setup complete!")
        print(f"📍 ChromaDB location: {chroma_path}")
        print(f"🔧 Environment variable set")
        print(f"\n📋 Next: Restart your Render service")
    else:
        print("\n💡 This script is designed to run on Render")
        print("   Upload it to your Render service and run it there") 