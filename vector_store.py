from langchain_core.embeddings import Embeddings
from langchain_qdrant import Qdrant
from qdrant_client import QdrantClient

def get_vector_store(client: QdrantClient, collection_name: str, embedding: Embeddings) -> Qdrant:
    vector_store = Qdrant(
        client=client,
        collection_name=collection_name,
        embeddings=embedding
    )
    return vector_store