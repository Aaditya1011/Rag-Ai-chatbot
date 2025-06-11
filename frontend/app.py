import streamlit as st
import requests

if "doc_id" not in st.session_state:
    st.session_state["doc_id"] = None
if "uploaded" not in st.session_state:
    st.session_state["uploaded"] = None

FASTAPI_URL = 'http://127.0.0.1:8000'

st.title("RAG Chatbot")

uploaded_files = st.file_uploader("Upload Document",type=["pdf","png","jpg","jpeg","txt"],accept_multiple_files=True)

if uploaded_files and not st.session_state["uploaded"]:
    if st.button("Upload"):

        # upload.
        with st.spinner("Uploading..."):
            files = [
                ("files", (file.name, file, file.type)) 
                for file in uploaded_files
            ]

            res = requests.post(f"{FASTAPI_URL}/upload/",files=files)
            doc_ids = []
            for i in res.json().get("files",[]):
                doc_ids.append(i["doc_id"])
            
            st.success(f"Files Uploaded.")
            st.session_state["doc_id"] = doc_ids
            st.session_state["uploaded"] = True
    
        # extract.
        with st.spinner("Extracting Text..."):
            extract = requests.post(f"{FASTAPI_URL}/extract-batch/",json=doc_ids)
            st.success("Text Extracted.")

        # Process.
        with st.spinner("Coverting to vectors..."):
            process = requests.post(f"{FASTAPI_URL}/process-batch",json=doc_ids)
            st.success("Converted to vectors & stored in VectorDB.")

# Ask.
questions = st.text_area("Ask questions ( one per line ): ")

if st.button("Ask"):
    question_list = [q.strip() for q in questions.split("\n") if q.strip()]

    if question_list and st.session_state["doc_id"]:
        with st.spinner("Thinking..."):

            res = requests.post(f"{FASTAPI_URL}/ask?top_k=5",json={"questions": question_list})

            if res.status_code == 200:
                for pair in res.json()["qa_pairs"]:
                    st.write(f"**Question**: {pair['question']}")
                    st.write(f"**Answer**: {pair['answer']}")
                    #st.write(f"**Citation**: {pair['citation']}")
                st.write(f"**Themes**: {res.json()['themes']}")
            else:
                st.error(f"Something went wrong. Status Code : {res.status_code}")




