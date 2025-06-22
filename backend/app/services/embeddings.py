from pathlib import Path
from typing import List, Tuple, Dict
import httpx
from app.config import TOGETHER_API_KEY
from fastapi import HTTPException
from app.services.text_extractor import extract_text
   
    
def load_extracted_text(doc_id:str):

    UPLOAD_DIR = Path("data/uploads")
    all_structured = []

    # find the file using doc_id.
    matched_files = list(UPLOAD_DIR.glob(f"{doc_id}.*"))

    # file path.
    file_path = matched_files[0]

    # extract text.
    try:
        pages = extract_text(str(file_path))
    except Exception as e:
        raise HTTPException(status_code=500,detail=f"Text extraction failed for {doc_id}: {e}")

    # meta_data like paragraph, lines.
    for page in pages:

        if not page["text"]:
            continue

        paragraphs = page["text"].split("\n\n")

        for p_idx, para in enumerate(paragraphs,start=1):
            lines = para.splitlines()
            for l_idx, line in enumerate(lines, start=1):
                all_structured.append({
                    "doc_id" : doc_id,
                    "page" : page["page_num"],
                    "paragraph" : p_idx,
                    "line" : l_idx,
                    "text" : line.strip()
                })

    return {"extracted" : all_structured}


def split_text_into_chunks_with_metadata(structured_data: List[Dict],chunk_size: int = 100,
overlap: int = 10) -> Tuple[List[str], List[Dict]]:
    """
    Splits structured text into overlapping chunks and returns both chunks and their metadata.
    """

    chunks = []
    metadata = []
    start = 0
    texts = [item["text"] for item in structured_data]


    while start < len(texts):
        end = start + chunk_size
        chunk_lines = structured_data[start:end]
        chunk_text = " ".join(line["text"] for line in chunk_lines)
        chunks.append(chunk_text)

        meta = {
            "doc_id": chunk_lines[0]["doc_id"],
            "page_range": str([chunk_lines[0]["page"], chunk_lines[-1]["page"]]),
            "paragraph_range": str([chunk_lines[0]["paragraph"], chunk_lines[-1]["paragraph"]]),
            "line_range": str([chunk_lines[0]["line"], chunk_lines[-1]["line"]]),
        }

        metadata.append(meta)

        start += chunk_size - overlap

    return chunks, metadata

async def get_embeddings_from_api(chunks:List[str]) -> List[List[float]]:
    """
    Get embeddings for each chunk using Together.AI.
    """

    # api url and key.
    API_URL = "https://api.together.xyz/v1/embeddings"
    API_KEY = TOGETHER_API_KEY
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

    