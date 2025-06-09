from pathlib import Path
from typing import List
import httpx
from app.config import together_api_key

def load_extracted_text(doc_id:str) -> str:
    """
    Load the extracted text for a given document ID from the uploads folder.
    """
    # get the text path or raise error if not found.
    try:
        text_path =  f"data/extracted/{doc_id}.txt"

        # return content as string.
        with open(text_path, "r", encoding="utf-8") as f:
            return f.read()

    except Exception as e:
        raise FileNotFoundError(f"Extracted text file for doc_id {doc_id} not found: {e}.")     
    

def split_text_into_chunks(text:str, chunk_size:int = 500, overlap:int = 50) -> List[str]:
    """
    Split the text into overlapping chunks to prepare for embedding.
    """

    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap

    return chunks

async def get_embeddings_from_api(chunks:List[str]) -> List[List[float]]:
    """
    Get embeddings for each chunk using Together.AI.
    """

    # api url and key.
    API_URL = "https://api.together.xyz/v1/embeddings"
    API_KEY = together_api_key
    MODEL = "togethercomputer/m2-bert-80M-8k-retrieval"

    # headers for api call.
    headers = {
        "Authorization" : f"Bearer {API_KEY}",
        "Content-Type" : "application/json"
    }

    # get embeddings from api.
    embeddings = []

    async with httpx.AsyncClient() as client:
        for chunk in chunks:
            # input for api call.
            payload = {
                "model" : MODEL,
                "input" : chunk
            }

            # response from api call.
            response = await client.post(API_URL, json=payload, headers=headers)

            # check if response is valid.
            if response.status_code == 200:
                embedding = response.json()["data"][0]["embedding"]
                embeddings.append(embedding)
            
            # otherwise throw error.
            else:
                raise Exception(f"Failed to get embedding: {response.text}")

    return embeddings

    