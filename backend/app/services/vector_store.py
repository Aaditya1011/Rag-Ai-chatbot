import chromadb
from chromadb.config import Settings
from typing import List

'''
from sentence_transformers import SentenceTransformer

class SentenceTransformerEmbedding:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def __call__(self, texts: List[str]) -> List[List[float]]:
        # Encode the list of texts into embeddings (list of float vectors)
        embeddings = self.model.encode(texts)
        return embeddings.tolist()
'''

# ChromaDB client Initialization.
client = chromadb.Client(Settings(
    persist_directory="backend/data/chroma",
    anonymized_telemetry=False
))


# Create or get collection.
collection = client.get_or_create_collection(name="document_chunks")


def store_chunks_with_embeddings(chunks: List[str], embeddings: List[List[float]], doc_id: str):
    """
    Stores text chunks with their embeddings in chromadb with doc_id as metadata.
    """

    # generate id for every chunk.
    ids = [f"{doc_id}_{i}" for i in range(len(chunks))]

    # if user uploads same file again remove chunks from previous file.
    existing_ids = collection.get(where={"doc_id": doc_id})["ids"]
    if existing_ids:
        collection.delete(ids=existing_ids)

    # doc_id and chunk id as metadata.
    metadata = [{"doc_id" : doc_id, "chunk_index" : i} for i in range(len(chunks))]

    # add to collection.
    collection.add(
        documents=chunks,
        embeddings=embeddings,
        ids=ids,
        metadatas=metadata
    )

# serach 
def search_similar_chunks(query_embedding: List[float], top_k: int = 5):
    """
    Search ChromaDB for chunks most similar to the query embedding.
    """

    # get results form ChromaDB.
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents","metadatas"]
    )

    # create metadata for matched chunks.
    matched_chunks = []
    for doc,meta in zip(results["documents"][0],results["metadatas"][0]):
        matched_chunks.append({
            "chunk" : doc,
            "doc_id" : meta["doc_id"],
            "chunk_index" : meta["chunk_index"]

        })

    return matched_chunks