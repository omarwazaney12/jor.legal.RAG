import requests
import os
from typing import List

class CustomOpenAIEmbeddings:
    """Custom OpenAI embeddings that bypasses the problematic client"""
    
    def __init__(self, model: str = "text-embedding-ada-002"):
        self.model = model
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not found")
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple documents"""
        embeddings = []
        for text in texts:
            response = requests.post(
                "https://api.openai.com/v1/embeddings",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={"input": text, "model": self.model}
            )
            response.raise_for_status()
            embedding = response.json()["data"][0]["embedding"]
            embeddings.append(embedding)
        return embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """Generate embedding for a single query"""
        return self.embed_documents([text])[0]

# Test the custom embeddings
if __name__ == "__main__":
    try:
        embedder = CustomOpenAIEmbeddings()
        result = embedder.embed_documents(["test document"])
        print(f"✅ Custom embeddings work! Dimension: {len(result[0])}")
    except Exception as e:
        print(f"❌ Custom embeddings failed: {e}") 