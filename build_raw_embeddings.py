import json
import os
from openai import OpenAI
from dotenv import load_dotenv
import time
import numpy as np

# Load environment variables
load_dotenv()

def build_raw_embeddings():
    """Build embeddings without ChromaDB - just OpenAI + JSON storage"""
    
    print("🧹 Starting raw embeddings build (no ChromaDB locally)...")
    
    # 1. Load source documents
    print("📖 Loading source documents...")
    with open('mit_jordan_data/mit_jordan_documents.json', 'r', encoding='utf-8') as f:
        documents_data = json.load(f)
    
    print(f"📊 Loaded {len(documents_data)} documents")
    
    # 2. Initialize OpenAI client
    print("🔑 Initializing OpenAI client...")
    openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    
    # 3. Process documents in batches
    batch_size = 10  # Small batches to avoid rate limits
    total_docs = len(documents_data)
    
    # Storage for all embeddings data
    all_embeddings_data = {
        'documents': [],
        'metadatas': [],
        'embeddings': [],
        'ids': [],
        'model': 'text-embedding-ada-002',
        'total_docs': total_docs,
        'embedding_dim': 1536  # OpenAI ada-002 dimension
    }
    
    print(f"🔄 Starting batch processing ({total_docs} docs in batches of {batch_size})...")
    
    for i in range(0, total_docs, batch_size):
        batch = documents_data[i:i + batch_size]
        batch_num = (i // batch_size) + 1
        total_batches = (total_docs + batch_size - 1) // batch_size
        
        print(f"\n🔄 Processing batch {batch_num}/{total_batches} ({len(batch)} documents)")
        
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
            
            # Truncate if too long (OpenAI has token limits)
            if len(doc_text) > 8000:  # Roughly 2000 tokens
                doc_text = doc_text[:8000] + "..."
            
            texts.append(doc_text.strip())
            metadatas.append({
                'title': doc.get('title', 'Unknown'),
                'url': doc.get('url', ''),
                'doc_type': doc.get('type', 'legal_document'),
                'batch': batch_num,
                'doc_index': i + j
            })
            ids.append(doc_id)
        
        # Get embeddings from OpenAI
        try:
            print(f"  🤖 Calling OpenAI API for {len(texts)} texts...")
            response = openai_client.embeddings.create(
                input=texts,
                model="text-embedding-ada-002"
            )
            embeddings = [data.embedding for data in response.data]
            
            # Store the data
            all_embeddings_data['documents'].extend(texts)
            all_embeddings_data['metadatas'].extend(metadatas)
            all_embeddings_data['embeddings'].extend(embeddings)
            all_embeddings_data['ids'].extend(ids)
            
            print(f"  ✅ Successfully processed batch {batch_num}")
            print(f"  📊 Total processed so far: {len(all_embeddings_data['documents'])}")
            
        except Exception as e:
            print(f"  ❌ Error processing batch {batch_num}: {e}")
            print(f"  ⏸️  Saving progress so far...")
            break
        
        # Rate limiting pause
        print(f"  ⏱️  Pausing 2 seconds for rate limiting...")
        time.sleep(2)
    
    # 4. Save embeddings data
    print(f"\n💾 Saving embeddings data...")
    
    # Create directory
    os.makedirs('./railway_embeddings_data', exist_ok=True)
    
    # Save as JSON (for loading in Railway)
    output_file = './railway_embeddings_data/embeddings_data.json'
    print(f"  📁 Writing to: {output_file}")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_embeddings_data, f, ensure_ascii=False, indent=2)
    
    file_size_mb = os.path.getsize(output_file) / 1024 / 1024
    print(f"  ✅ Saved embeddings data")
    print(f"  📊 File size: {file_size_mb:.1f} MB")
    print(f"  📄 Documents embedded: {len(all_embeddings_data['documents'])}")
    
    # 5. Create test/verification script
    print(f"\n🔧 Creating verification script...")
    
    verification_script = '''import json
import numpy as np

def test_embeddings(data_file="./railway_embeddings_data/embeddings_data.json"):
    """Test the raw embeddings data"""
    
    print("🔍 Loading and testing embeddings...")
    
    with open(data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"📊 Loaded {len(data['documents'])} documents")
    print(f"🎯 Model: {data['model']}")
    print(f"📐 Embedding dimension: {data['embedding_dim']}")
    
    # Test embedding dimensions
    embeddings = np.array(data['embeddings'])
    print(f"🔢 Embeddings shape: {embeddings.shape}")
    
    # Show some sample documents
    print(f"\\n📄 Sample documents:")
    for i in range(min(3, len(data['documents']))):
        title = data['metadatas'][i].get('title', 'No title')
        print(f"  {i+1}. {title[:80]}...")
    
    print(f"\\n🎉 Embeddings data is valid and ready for Railway!")

if __name__ == "__main__":
    test_embeddings()
'''
    
    with open('./railway_embeddings_data/verify_embeddings.py', 'w') as f:
        f.write(verification_script)
    
    # 6. Create Railway loader script
    print(f"🚀 Creating Railway loader script...")
    
    railway_loader = '''import json
import chromadb
import os

def load_prebuilt_embeddings():
    """Load pre-built embeddings into ChromaDB on Railway"""
    
    print("📖 Loading pre-built embeddings for Railway...")
    
    # Load the embeddings data
    data_file = "./railway_embeddings_data/embeddings_data.json"
    if not os.path.exists(data_file):
        print(f"❌ Embeddings file not found: {data_file}")
        return None, None
    
    with open(data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"📊 Loaded {len(data['documents'])} documents")
    
    # Create ChromaDB client (should work on Railway)
    client = chromadb.Client()  # In-memory for Railway
    
    # Create collection
    collection = client.create_collection(
        name="jordanian_legal_docs",
        metadata={
            "description": "Jordanian legal documents for RAG system",
            "model": data.get('model', 'text-embedding-ada-002'),
            "total_docs": data.get('total_docs', len(data['documents']))
        }
    )
    
    # Add all data at once
    print("🔄 Adding documents to ChromaDB...")
    collection.add(
        documents=data['documents'],
        metadatas=data['metadatas'],
        embeddings=data['embeddings'],
        ids=data['ids']
    )
    
    final_count = collection.count()
    print(f"✅ Loaded {final_count} documents into ChromaDB")
    
    # Test query
    test_results = collection.query(
        query_texts=["banking regulations"],
        n_results=3
    )
    print(f"🔍 Test query returned {len(test_results['documents'][0])} results")
    
    return client, collection

if __name__ == "__main__":
    client, collection = load_prebuilt_embeddings()
    if collection:
        print(f"🎉 ChromaDB ready with {collection.count()} documents!")
'''
    
    with open('./railway_embeddings_data/railway_loader.py', 'w') as f:
        f.write(railway_loader)
    
    print(f"✅ Created Railway loader: ./railway_embeddings_data/railway_loader.py")
    
    print(f"\n🎉 EMBEDDING BUILD COMPLETE!")
    print(f"📁 Files created:")
    print(f"  - embeddings_data.json ({file_size_mb:.1f} MB)")
    print(f"  - verify_embeddings.py (test locally)")
    print(f"  - railway_loader.py (use on Railway)")
    print(f"\n🚀 Next steps:")
    print(f"  1. Run: python railway_embeddings_data/verify_embeddings.py")
    print(f"  2. Deploy to Railway with this data!")

if __name__ == "__main__":
    # Check for OpenAI API key
    if not os.getenv('OPENAI_API_KEY'):
        print("❌ Please set your OPENAI_API_KEY in .env file")
        exit(1)
    
    build_raw_embeddings() 