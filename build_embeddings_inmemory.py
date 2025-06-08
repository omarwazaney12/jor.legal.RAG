import json
import os
import shutil
import chromadb
from openai import OpenAI
from dotenv import load_dotenv
import time
import pickle

# Load environment variables
load_dotenv()

def build_embeddings_inmemory():
    """Build embeddings using in-memory ChromaDB to avoid SQLite issues"""
    
    print("🧹 Starting in-memory ChromaDB build...")
    
    # 1. Create in-memory ChromaDB client (no persistence issues)
    print("📁 Creating in-memory ChromaDB client...")
    client = chromadb.Client()  # In-memory client
    
    # 2. Create collection
    print("📋 Creating collection...")
    collection = client.create_collection(
        name="jordanian_legal_docs",
        metadata={"description": "Jordanian legal documents for RAG system"}
    )
    
    # 3. Load source documents
    print("📖 Loading source documents...")
    with open('mit_jordan_data/mit_jordan_documents.json', 'r', encoding='utf-8') as f:
        documents_data = json.load(f)
    
    print(f"📊 Loaded {len(documents_data)} documents")
    
    # 4. Initialize OpenAI client
    openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    
    # 5. Process documents in batches
    batch_size = 10  # Small batches to avoid rate limits
    total_docs = len(documents_data)
    
    all_embeddings_data = {
        'documents': [],
        'metadatas': [],
        'embeddings': [],
        'ids': []
    }
    
    for i in range(0, total_docs, batch_size):
        batch = documents_data[i:i + batch_size]
        batch_num = (i // batch_size) + 1
        total_batches = (total_docs + batch_size - 1) // batch_size
        
        print(f"🔄 Processing batch {batch_num}/{total_batches} ({len(batch)} documents)")
        
        # Prepare batch data
        texts = []
        metadatas = []
        ids = []
        
        for j, doc in enumerate(batch):
            doc_id = f"doc_{i + j}"
            
            # Create comprehensive text for embedding
            doc_text = ""
            if 'title' in doc:
                doc_text += f"Title: {doc['title']}\n"
            if 'content' in doc:
                doc_text += f"Content: {doc['content']}\n"
            if 'url' in doc:
                doc_text += f"Source: {doc['url']}\n"
            
            texts.append(doc_text.strip())
            metadatas.append({
                'title': doc.get('title', 'Unknown'),
                'url': doc.get('url', ''),
                'doc_type': doc.get('type', 'legal_document'),
                'batch': batch_num
            })
            ids.append(doc_id)
        
        # Get embeddings from OpenAI
        try:
            response = openai_client.embeddings.create(
                input=texts,
                model="text-embedding-ada-002"
            )
            embeddings = [data.embedding for data in response.data]
            
            # Add to in-memory collection
            collection.add(
                documents=texts,
                metadatas=metadatas,
                embeddings=embeddings,
                ids=ids
            )
            
            # Store for manual saving
            all_embeddings_data['documents'].extend(texts)
            all_embeddings_data['metadatas'].extend(metadatas)
            all_embeddings_data['embeddings'].extend(embeddings)
            all_embeddings_data['ids'].extend(ids)
            
            print(f"✅ Added batch {batch_num} to ChromaDB")
            
        except Exception as e:
            print(f"❌ Error processing batch {batch_num}: {e}")
            break
        
        # Rate limiting pause
        time.sleep(1)
    
    # 6. Test the in-memory collection
    final_count = collection.count()
    print(f"\n🎉 In-memory embedding complete!")
    print(f"📊 Total documents embedded: {final_count}")
    
    # 7. Test a sample query
    print("\n🔍 Testing sample query...")
    try:
        results = collection.query(
            query_texts=["What are the banking regulations in Jordan?"],
            n_results=3
        )
        print(f"✅ Sample query returned {len(results['documents'][0])} results")
        print(f"📄 First result title: {results['metadatas'][0][0].get('title', 'N/A')}")
    except Exception as e:
        print(f"⚠️  Test query failed: {e}")
    
    # 8. Save embeddings data for Railway
    print("\n💾 Saving embeddings data for Railway...")
    
    # Create directory
    os.makedirs('./railway_embeddings_data', exist_ok=True)
    
    # Save as JSON (for loading in Railway)
    with open('./railway_embeddings_data/embeddings_data.json', 'w', encoding='utf-8') as f:
        json.dump(all_embeddings_data, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Saved embeddings data to: ./railway_embeddings_data/embeddings_data.json")
    print(f"📁 Size: {os.path.getsize('./railway_embeddings_data/embeddings_data.json') / 1024 / 1024:.1f} MB")
    
    # 9. Create Railway-compatible loader script
    loader_script = '''import json
import chromadb
from typing import List, Dict, Any

def load_prebuilt_embeddings(data_file: str = "./railway_embeddings_data/embeddings_data.json"):
    """Load pre-built embeddings into ChromaDB on Railway"""
    
    print("📖 Loading pre-built embeddings...")
    
    # Load the embeddings data
    with open(data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Create in-memory ChromaDB client
    client = chromadb.Client()
    
    # Create collection
    collection = client.create_collection(
        name="jordanian_legal_docs",
        metadata={"description": "Jordanian legal documents for RAG system"}
    )
    
    # Add all data at once
    collection.add(
        documents=data['documents'],
        metadatas=data['metadatas'],
        embeddings=data['embeddings'],
        ids=data['ids']
    )
    
    print(f"✅ Loaded {len(data['documents'])} documents into ChromaDB")
    return client, collection

if __name__ == "__main__":
    client, collection = load_prebuilt_embeddings()
    print(f"🎉 Collection ready with {collection.count()} documents")
'''
    
    with open('./railway_embeddings_data/load_embeddings.py', 'w') as f:
        f.write(loader_script)
    
    print(f"✅ Created loader script: ./railway_embeddings_data/load_embeddings.py")
    print(f"\n🚀 Ready for Railway deployment!")
    print(f"📁 Upload the entire 'railway_embeddings_data' folder to Railway")

if __name__ == "__main__":
    # Check for OpenAI API key
    if not os.getenv('OPENAI_API_KEY'):
        print("❌ Please set your OPENAI_API_KEY in .env file")
        exit(1)
    
    build_embeddings_inmemory() 