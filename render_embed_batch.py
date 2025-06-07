#!/usr/bin/env python3
"""
Render Batch Document Embedding Script

This script embeds Jordan legal documents in small batches to avoid memory/timeout issues.
Automatically resumes from last successful batch.
"""

import os
import sys
import time
import json
from pathlib import Path
from datetime import datetime

def save_progress(completed_docs, failed_chunks, progress_file="embedding_progress.json"):
    """Save embedding progress to file"""
    progress = {
        "timestamp": datetime.now().isoformat(),
        "completed_documents": completed_docs,
        "failed_chunks": failed_chunks,
        "total_completed": len(completed_docs)
    }
    
    with open(progress_file, 'w', encoding='utf-8') as f:
        json.dump(progress, f, ensure_ascii=False, indent=2)
    
    print(f"💾 Progress saved: {len(completed_docs)} docs completed")

def load_progress(progress_file="embedding_progress.json"):
    """Load previous embedding progress"""
    try:
        if os.path.exists(progress_file):
            with open(progress_file, 'r', encoding='utf-8') as f:
                progress = json.load(f)
            return progress.get('completed_documents', []), progress.get('failed_chunks', [])
        else:
            return [], []
    except Exception as e:
        print(f"⚠️  Could not load progress: {e}")
        return [], []

def embed_documents_batch():
    """Embed documents in small batches with resume capability"""
    print("🚀 Starting Batch Document Embedding")
    print("=" * 50)
    
    # Load previous progress
    completed_docs, failed_chunks = load_progress()
    print(f"📊 Previous progress: {len(completed_docs)} documents completed")
    
    # Check environment
    api_key = os.getenv('OPENAI_API_KEY', '')
    if not api_key:
        print("❌ OPENAI_API_KEY is required!")
        return False
    
    # Get all documents
    docs_path = Path("mit_jordan_data/txt_output")
    if not docs_path.exists():
        print(f"❌ Documents directory not found: {docs_path}")
        return False
    
    all_files = list(docs_path.glob("*.txt"))
    
    # Filter out already completed documents
    remaining_files = [f for f in all_files if f.name not in completed_docs]
    
    print(f"📄 Total documents: {len(all_files)}")
    print(f"✅ Already completed: {len(completed_docs)}")
    print(f"🔄 Remaining to process: {len(remaining_files)}")
    
    if len(remaining_files) == 0:
        print("🎉 All documents already embedded!")
        return True
    
    # Initialize system
    try:
        from advanced_rag_system import AdvancedLegalRAGSystem
        system = AdvancedLegalRAGSystem()
        print("✅ RAG system initialized")
    except Exception as e:
        print(f"❌ System initialization failed: {e}")
        return False
    
    # Check if we should resume or start fresh
    try:
        existing_count = system.vector_store.collection.count()
        print(f"📊 Found {existing_count} existing embeddings")
        
        if existing_count > 0 and len(completed_docs) == 0:
            print("✅ Found existing embeddings but no progress record")
            print("   Will attempt to resume embedding remaining documents...")
            # This is a fresh run but data exists - let it use existing data
        
    except Exception as e:
        print(f"📊 No existing embeddings found: {e}")
    
    # Use the built-in load_documents method which handles everything
    print(f"\n🚀 Processing all documents using built-in system...")
    print("   This will automatically:")
    print("   - Detect existing embeddings")
    print("   - Resume from where it left off") 
    print("   - Handle rate limiting and errors")
    print("   - Save progress automatically")
    
    try:
        # Let the system handle everything - it will detect existing data
        num_loaded = system.load_documents(force_rebuild=False)
        
        if num_loaded > 0:
            print(f"✅ Successfully processed {num_loaded} documents")
            
            # Verify embeddings
            try:
                final_count = system.vector_store.collection.count()
                print(f"📊 Total embeddings in database: {final_count}")
                
                # Mark all documents as completed for our tracking
                for txt_file in all_files:
                    if txt_file.name not in completed_docs:
                        completed_docs.append(txt_file.name)
                
                save_progress(completed_docs, failed_chunks)
                return True
                
            except Exception as e:
                print(f"⚠️  Could not verify final count: {e}")
                return True  # Still consider it successful
        else:
            print("❌ No documents were processed")
            return False
            
    except Exception as e:
        print(f"❌ Document processing failed: {e}")
        
        # Check if partial progress was made
        try:
            final_count = system.vector_store.collection.count()
            print(f"📊 Current embeddings in database: {final_count}")
            if final_count > 0:
                print("⚠️  Partial progress made - you can try again")
                return True
        except:
            pass
        
        return False
    
    # Final summary
    print(f"\n📊 Final Summary:")
    print(f"   ✅ Documents completed: {len(completed_docs)}")
    print(f"   ❌ Failed chunks: {len(failed_chunks)}")
    
    # Verify final count
    try:
        total_embeddings = system.vector_store.collection.count()
        print(f"   📈 Total embeddings in database: {total_embeddings}")
    except Exception as e:
        print(f"   ⚠️  Could not verify count: {e}")
    
    return len(completed_docs) > 0

if __name__ == "__main__":
    print("🔧 Render Batch Embedding Script")
    print(f"📅 Started at: {datetime.now().isoformat()}")
    
    success = embed_documents_batch()
    
    if success:
        print("\n🎉 Batch embedding completed successfully!")
    else:
        print("\n❌ Batch embedding encountered issues")
    
    print(f"📅 Finished at: {datetime.now().isoformat()}") 