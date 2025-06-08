import json
import os
import shutil
import chromadb
from chromadb.config import Settings
from openai import OpenAI
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()

def clean_and_build_embeddings():
    """Build fresh ChromaDB embeddings from source JSON data"""
    
    print("🧹 Starting fresh ChromaDB build...")
    
    # 1. Clean up old ChromaDB directories
    chroma_dirs = ['./railway_chroma_db', './chroma_db_render', './test_chroma_db']
    for dir_path in chroma_dirs:
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)
            print(f"🗑️  Removed old directory: {dir_path}")
    
    # 2. Create fresh ChromaDB client with DuckDB backend
    print("📁 Creating fresh ChromaDB client with DuckDB backend...")
    settings = Settings(
        is_persistent=True,
        persist_directory="./railway_chroma_db",
        anonymized_telemetry=False
    )
    
    client = chromadb.PersistentClient(
        path='./railway_chroma_db',
        settings=settings
    )
    
    # 3. Create collection with OpenAI embeddings
    print("📋 Creating collection...")
    collection = client.create_collection(
        name="jordanian_legal_docs",
        metadata={"description": "Jordanian legal documents for RAG system"}
    )
    
    # 4. Load source documents
    print("📖 Loading source documents...")
    with open('mit_jordan_data/mit_jordan_documents.json', 'r', encoding='utf-8') as f:
        documents_data = json.load(f)
    
    print(f"📊 Loaded {len(documents_data)} documents")
    
    # 5. Initialize OpenAI client
    openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    
    # 6. Process documents in batches
    batch_size = 10  # Small batches to avoid rate limits
    total_docs = len(documents_data)
    
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
            
            # Add to ChromaDB collection
            collection.add(
                documents=texts,
                metadatas=metadatas,
                embeddings=embeddings,
                ids=ids
            )
            
            print(f"✅ Added batch {batch_num} to ChromaDB")
            
        except Exception as e:
            print(f"❌ Error processing batch {batch_num}: {e}")
            break
        
        # Rate limiting pause
        time.sleep(1)
    
    # 7. Verify the collection
    final_count = collection.count()
    print(f"\n🎉 Embedding complete!")
    print(f"📊 Total documents embedded: {final_count}")
    
    # 8. Test a sample query
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
    
    print(f"\n🚀 Ready for Railway deployment!")
    print(f"📁 ChromaDB saved to: ./railway_chroma_db/")

if __name__ == "__main__":
    # Check for OpenAI API key
    if not os.getenv('OPENAI_API_KEY'):
        print("❌ Please set your OPENAI_API_KEY in .env file")
        exit(1)
    
    clean_and_build_embeddings() 