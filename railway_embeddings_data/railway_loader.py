import json
import chromadb
import os

def load_prebuilt_embeddings():
    """Load pre-built embeddings into ChromaDB on Railway"""
    
    print("Loading pre-built embeddings for Railway...")
    
    # Load the embeddings data
    data_file = "./railway_embeddings_data/embeddings_data.json"
    if not os.path.exists(data_file):
        print(f"Embeddings file not found: {data_file}")
        return None, None
    
    with open(data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Loaded {len(data['documents'])} documents")
    
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
    print("Adding documents to ChromaDB...")
    collection.add(
        documents=data['documents'],
        metadatas=data['metadatas'],
        embeddings=data['embeddings'],
        ids=data['ids']
    )
    
    final_count = collection.count()
    print(f"Loaded {final_count} documents into ChromaDB")
    
    # Test query
    test_results = collection.query(
        query_texts=["banking regulations"],
        n_results=3
    )
    print(f"Test query returned {len(test_results['documents'][0])} results")
    
    return client, collection

if __name__ == "__main__":
    client, collection = load_prebuilt_embeddings()
    if collection:
        print(f"ChromaDB ready with {collection.count()} documents!") 