from typing import List, Dict
import chromadb
import os

os.makedirs("data/chroma",exist_ok=True)
client = chromadb.PersistentClient(path="data/chroma")


def store_chunks_with_embeddings(chunks: List[str],embeddings: List[List[float]],metadata: List[Dict]):

    print("store chunks with embeddings called.")
    collection =  client.get_or_create_collection(name="document_chunks",embedding_function=None)

    ids = [f"{meta['doc_id']}_{i}" for i, meta in enumerate(metadata)]

    collection.add(
        documents=chunks,
        embeddings=embeddings,
        ids=ids,
        metadatas=metadata
    )

# search.
def search_similar_chunks(query_embedding: List[float], top_k: int = 5,client=client):
    """
    Search ChromaDB for chunks most similar to the query embedding.
    """
    
    # initialize collection.
    collection =  client.get_or_create_collection(name="document_chunks",embedding_function=None)

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
            "paragraph_range" : meta['paragraph_range'],
            "page_range" : meta["page_range"],
            "line_range" : meta["line_range"]
        })

    return matched_chunks