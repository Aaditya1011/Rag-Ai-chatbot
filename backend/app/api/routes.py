from fastapi import APIRouter, File, UploadFile, HTTPException, Body
from pathlib import Path
from datetime import datetime
from typing import List
import os
import uuid
from app.services.text_extractor import extract_text
from app.services.llm_api import get_answer_from_llm, identify_themes_in_responses
from app.services.embeddings import load_extracted_text
#from app.services.embeddings import split_text_into_chunks
from app.services.embeddings import get_embeddings_from_api
from app.services.vector_store import store_chunks_with_embeddings

from app.services.embeddings import split_text_into_chunks_with_metadata

from chromadb import Client
from chromadb.config import Settings

# from fastapi.responses import JSONResponse

# router initialisation.
router = APIRouter()

# path for uploaded files (create if not exists).
UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True,exist_ok=True)

# file upload route/api.
@router.post("/upload/")
async def upload_document(files: List[UploadFile] = File(...)):

    responses = []

    # iterate over all files.
    for file in files:
        
        # check for extension.
        ext = os.path.splitext(file.filename)[1].lower()
        if ext not in [".pdf",".txt",".png",".jpg",".jpeg"]:
            raise HTTPException(status_code=400,detail="Unsupported File Type.")

        # generation of unique id.
        doc_id = str(uuid.uuid4())
        file_name = f"{doc_id}{ext}"
        file_path = os.path.join(UPLOAD_DIR,file_name)

        # save file in data/uploads.
        with open(file_path,"wb") as f:
            content = await file.read()
            f.write(content)

        # add to responses.
        responses.append({
            "doc_id": doc_id,
            "original_filename": file.filename,
            "stored_filename": file_name
        })

    return {
        "status": "Upload Success",
        "upload_time": datetime.now().isoformat(),
        "files": responses
    }


# extract text route/api.
@router.post('/extract-batch/')
def extract_document_text(doc_ids: List[str]):

    all_structured = []

    for doc_id in doc_ids:
            
        # find the file using doc_id.
        matched_files = list(UPLOAD_DIR.glob(f"{doc_id}.*"))
        if not matched_files:
            continue

        # file path.
        file_path = matched_files[0]

        # extract text.
        try:
            pages = extract_text(str(file_path))
        except Exception as e:
            raise HTTPException(status_code=500,detail=f"Text extraction failed for {doc_id}: {e}")

        # meta_data like paragraph, lines.
        full_text = ""

        for page in pages:

            if not page["text"]:
                continue

            paragraphs = page["text"].split("\n\n")
            full_text += page["text"].strip()

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

        # save the text in txt file.
        extracted_path = Path("data/extracted")
        extracted_path.mkdir(parents=True,exist_ok=True)

        with open(extracted_path / f"{doc_id}.txt", "w", encoding="utf-8") as f:
            f.write(full_text.strip())

    return {"extracted" : all_structured}

@router.post("/process-batch")
async def process_batch(doc_ids: List[str]):
    """
    Processes a list of document ids :
        - loads extracted text.
        - splits into chunks.
        - gets embeddings from together.ai.
        - stores in chromaDB.
    """

    processed_docs = []

    for doc_id in doc_ids:
        
        try:
            # load full text from file.
            structured_data = load_extracted_text(doc_id)
            print("text loaded")

            # split the text into chunks.
            # chunks = split_text_into_chunks(text)
            chunks,metadata = split_text_into_chunks_with_metadata(structured_data["extracted"])
            print("chunks loaded")

            if not chunks:
                continue

            # get the embeddings.
            embeddings = await get_embeddings_from_api(chunks)
            print("embeddings received.")

            # store in chromaDB.
            # store_chunks_with_embeddings(chunks, embeddings, doc_id)
            store_chunks_with_embeddings(chunks, embeddings, metadata)

            processed_docs.append(doc_id)
            print("docs appended.")
        
        except Exception as e:
            raise HTTPException(status_code=500,detail=f"Failed to process {doc_id}: {str(e)}")

    return {
        "status":f"Processed {len(processed_docs)} documents successfully.",
        "doc_ids": processed_docs
    }


@router.post("/ask")
async def ask_question(questions: List[str] = Body(...,embed=True), top_k: int = 5):
    """
    Takes a question, finds similar chunks, and returns an LLM generated answer.
    """

    answers = []
    citation = []

    # get answer for each question.
    for question in questions:
        ans,ans_meta = await get_answer_from_llm(question,top_k)
        answers.append(ans)
        citation.append(ans_meta)
    
    print('citation: ',citation)
    # identify themes in answers.
    themes = await identify_themes_in_responses(answers)

    return {
        "qa_pairs" : [{"question": q, "answer": a, "citation": c} for q,a,c in zip(questions,answers,citation)],
        "themes" : themes
    }
