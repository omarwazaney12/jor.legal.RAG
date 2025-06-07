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
    
    # Process documents in batches of 5
    BATCH_SIZE = 5
    batch_count = 0
    total_batches = (len(remaining_files) + BATCH_SIZE - 1) // BATCH_SIZE
    
    for i in range(0, len(remaining_files), BATCH_SIZE):
        batch_count += 1
        batch_files = remaining_files[i:i+BATCH_SIZE]
        
        print(f"\n📦 Processing Batch {batch_count}/{total_batches}")
        print(f"   Files: {[f.name for f in batch_files]}")
        
        batch_success = True
        
        for doc_file in batch_files:
            try:
                print(f"   🔄 Processing: {doc_file.name}")
                
                # Process single document
                with open(doc_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                if len(content.strip()) < 50:  # Skip very short documents
                    print(f"   ⚠️  Skipping short document: {doc_file.name}")
                    completed_docs.append(doc_file.name)
                    continue
                
                # Add to embeddings (this will automatically chunk and embed)
                try:
                    # Create a mini-batch with just this document
                    temp_docs = [{"content": content, "metadata": {"source": doc_file.name}}]
                    chunks = system._create_chunks(temp_docs)
                    
                    # Embed chunks with retry logic
                    success_count = 0
                    for chunk in chunks:
                        try:
                            embedding = system.embedding_model.embed_query(chunk.page_content)
                            
                            # Add to vector store
                            system.vector_store.add_texts(
                                texts=[chunk.page_content],
                                embeddings=[embedding],
                                metadatas=[chunk.metadata]
                            )
                            success_count += 1
                            
                            # Rate limiting - small delay
                            time.sleep(0.1)
                            
                        except Exception as chunk_error:
                            print(f"     ⚠️  Chunk failed: {str(chunk_error)[:100]}")
                            failed_chunks.append({
                                "document": doc_file.name,
                                "error": str(chunk_error),
                                "content_preview": chunk.page_content[:100]
                            })
                    
                    print(f"   ✅ {success_count}/{len(chunks)} chunks embedded")
                    completed_docs.append(doc_file.name)
                    
                except Exception as doc_error:
                    print(f"   ❌ Document failed: {doc_error}")
                    batch_success = False
                    break
                
                # Small delay between documents
                time.sleep(0.5)
                
            except Exception as e:
                print(f"   ❌ Error processing {doc_file.name}: {e}")
                batch_success = False
                break
        
        # Save progress after each batch
        save_progress(completed_docs, failed_chunks)
        
        if not batch_success:
            print(f"⚠️  Batch {batch_count} had errors - progress saved")
            break
        
        # Longer delay between batches to avoid rate limiting
        if batch_count < total_batches:
            print(f"   ⏱️  Waiting 30 seconds before next batch...")
            time.sleep(30)
    
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