import httpx
from typing import List, Dict, Tuple
from app.services.vector_store import search_similar_chunks
from app.services.embeddings import get_embeddings_from_api
from app.config import TOGETHER_API_KEY

# LLM API URL and KEY.
API_URL = "https://api.together.xyz/v1/chat/completions"
API_KEY = TOGETHER_API_KEY
MODEL = "mistralai/Mixtral-8x7B-Instruct-v0.1"

# get answer from the LLM.
async def get_answer_from_llm(query: str, top_k: int = 5) -> Tuple[str,List[Dict]]:
    """
    Get an answer for the query from the LLM using most relevant chunks.
    """

    # Get the query's embedding and similar chunks.
    query_embeddings = await get_embeddings_from_api([query])
    similar_chunks = search_similar_chunks(query_embeddings[0],top_k)

    # metadata of chunks used to form context.
    citation = []
    for i in similar_chunks:
        citation.append({
            "chunk" : i.get("chunk"),
            "doc_id": i.get("doc_id"),
            "paragraph_range": i.get("paragraph_range"),
            "page_range": i.get("page_range"),
            "line_range": i.get("line_range"),
        })


    # combine the matched chunks to from context for LLM.
    context = "\n\n".join([chunk["chunk"] for chunk in similar_chunks])

    # Input for LLM API.
    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant that answers questions using the given context."
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion:\n{query}"
            }
        ],
        "max_tokens": 300,
        "temperature": 0.2
    }
    
    # required headers.
    headers = {
        "Authorization" : f"Bearer {API_KEY}",
        "Content-Type" : "application/json"
    }

    # LLM API call.
    timeout = httpx.Timeout(60.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(API_URL, json=payload, headers=headers)

    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"], citation
    else:
        raise Exception(f"Error fetching answer from LLM API: {response.text}")


async def identify_themes_in_responses(responses: List[str]) -> Dict[str,List[str]]:
    """
    Identify and group response into themes based on LLM analysis.
    """

    user_content = "Identify the main themes in the following text snippets and group them accordingly.\n\n"

    for idx, response in enumerate(responses, 1):
        user_content += f"Snippet {idx}: {response}\n"

    user_content += "\nReturn the themes clearly, possibly as a dictionary or bullet points."

    # Input for LLM API.
    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "system",
                "content": "You are an expert language model that analyzes and groups text snippets into themes."
            },
            {
                "role": "user",
                "content": user_content
            }
        ],
        "max_tokens": 300,
        "temperature": 0.3
    }

    headers = {
        "Authorization" : f"Bearer {API_KEY}",
        "Content-Type" : "application/json"
    }
    
    # LLM API call.
    async with httpx.AsyncClient() as client:
        response = await client.post(API_URL, json=payload, headers=headers)

    if response.status_code == 200:
        themes = response.json()["choices"][0]["message"]["content"]
        return themes
    else:
        raise Exception(f"Error fetching themes from LLM API : {response.text}")
    