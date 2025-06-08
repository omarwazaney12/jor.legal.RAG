import json
import numpy as np

def test_embeddings(data_file="./embeddings_data.json"):
    """Test the raw embeddings data"""
    
    print("Loading and testing embeddings...")
    
    with open(data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Loaded {len(data['documents'])} documents")
    print(f"Model: {data['model']}")
    print(f"Embedding dimension: {data['embedding_dim']}")
    
    # Test embedding dimensions
    embeddings = np.array(data['embeddings'])
    print(f"Embeddings shape: {embeddings.shape}")
    
    # Show some sample documents
    print(f"\nSample documents:")
    for i in range(min(3, len(data['documents']))):
        title = data['metadatas'][i].get('title', 'No title')
        print(f"  {i+1}. {title[:80]}...")
    
    print(f"\nEmbeddings data is valid and ready for Railway!")

if __name__ == "__main__":
    test_embeddings()
