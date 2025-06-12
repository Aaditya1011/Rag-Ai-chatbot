import streamlit as st
import pandas as pd
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

            # store in session.
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

                try:
                    data = res.json()
                except requests.exceptions.JSONDecodeError:
                    st.error("Response was not valid JSON.")
                    st.stop()

                for qa in data["qa_pairs"]:
                    st.subheader("Question:" )
                    st.write(qa["question"])

                    st.subheader("Answer:" )
                    st.write(qa["answer"])
                    
                    if qa["citation"]:
                        citation_df = pd.DataFrame([
                            {
                                "Doc_ID" : c.get("doc_id"),
                                "Chunk" : c.get("chunk"),
                                "Paragraph" : c.get("pragraph_range"),
                                "Page" : c.get("page_range"),
                                "Line" : c.get("line_range"),
                            }

                            for c in qa["citation"]
                        ])

                        st.subheader("Citations:")
                        st.table(citation_df)
                    else:
                        st.write("No Citation Found.")
                    
                st.write(data["themes"])
            else:
                st.error(f"Something went wrong. Status Code : {res.status_code}")




