import chromadb

def check_chromadb_data():
    print("🔍 Checking ChromaDB data...")
    
    # Check chroma_db_render directory
    try:
        client = chromadb.PersistentClient(path='./chroma_db_render')
        collections = client.list_collections()
        print(f"\n📁 chroma_db_render directory:")
        if collections:
            for collection in collections:
                count = collection.count()
                print(f"  - Collection: {collection.name} ({count} documents)")
        else:
            print("  - No collections found")
    except Exception as e:
        print(f"  - Error accessing chroma_db_render: {e}")
    
    # Check test_chroma_db directory
    try:
        client = chromadb.PersistentClient(path='./test_chroma_db')
        collections = client.list_collections()
        print(f"\n📁 test_chroma_db directory:")
        if collections:
            for collection in collections:
                count = collection.count()
                print(f"  - Collection: {collection.name} ({count} documents)")
        else:
            print("  - No collections found")
    except Exception as e:
        print(f"  - Error accessing test_chroma_db: {e}")
    
    # Check main chroma_db directory
    try:
        client = chromadb.PersistentClient(path='./chroma_db')
        collections = client.list_collections()
        print(f"\n📁 chroma_db directory:")
        if collections:
            for collection in collections:
                count = collection.count()
                print(f"  - Collection: {collection.name} ({count} documents)")
        else:
            print("  - No collections found")
    except Exception as e:
        print(f"  - Error accessing chroma_db: {e}")

if __name__ == "__main__":
    check_chromadb_data() 