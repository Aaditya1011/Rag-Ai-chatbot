# 🧠 RAG-based AI Chatbot

A **Retrieval-Augmented Generation (RAG)** based AI chatbot that can understand user queries and respond with contextually relevant answers by searching from a custom collection of documents (PDFs, text files, images, etc.). The response also includes **document metadata like filename, page number, paragraph, and line references**.

---

## 📌 Features

- Upload and index 75+ documents (PDF, text, image with OCR)
- Semantic search using vector embeddings
- Context-aware responses using LLMs
- RAG architecture for combining retrieval and generation
- Output includes source reference (doc ID, page, paragraph, line)
- API backend using **FastAPI**
- Frontend using **Streamlit**

---

## 🛠️ Tech Stack

| Component     | Tool/Library                      |
|---------------|-----------------------------------|
| Backend       | FastAPI                           |
| Frontend      | Streamlit                         |
| Vector Store  | FAISS / Chroma                    |
| Embedding     | SentenceTransformers / OpenAI     |
| LLM Provider  | Together.ai API                   |
| OCR (Image)   | Tesseract / EasyOCR               |
| PDF Parsing   | PyMuPDF / PDFMiner / pdfplumber   |
| Data Formats  | PDF, TXT, JPG, PNG                |

---

## 🔁 Workflow Overview

### 1. **Document Upload & Preprocessing**
- Users upload documents through the UI.
- Documents are processed based on type:
  - **PDF/Text**: Split into paragraphs and lines.
  - **Image**: OCR is performed to extract text.
- Each piece of text is assigned a `doc_id`, `page number`, `paragraph`, and `line number`.

### 2. **Embedding & Vector Storage**
- Text chunks are converted into embeddings using a model like `sentence-transformers/all-MiniLM-L6-v2`.
- Embeddings are stored in a vector store like **FAISS** or **ChromaDB**, along with their metadata.

### 3. **Query Handling**
- User submits a query through the chatbot interface.
- The query is embedded and compared against stored vectors to find the top-k most relevant text chunks.
- These chunks are fetched with metadata.

### 4. **RAG Pipeline**
- Retrieved chunks are passed along with the user query to the LLM (e.g., Together.ai).
- The LLM generates a response using both the query and the context.
- The final response is returned **with cited sources** (doc, page, paragraph, line).

### 5. **Display Answer**
- The answer is shown in the chat interface.
- References to source documents are highlighted below the response.

---

## 🧪 Example

**Query**: *"What is the recommended dosage of medication X?"*

**Response**:
> The recommended dosage of medication X is 500mg twice daily, as noted in the prescription guidelines.

**References**:
- `Doc: medical_guide.pdf | Page: 12 | Paragraph: 3 | Line: 5`

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/Aaditya1011/rag-ai-chatbot.git
cd rag-ai-chatbot
```

### 2. Create Virtual Envirnoment.
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
pip install -r frontend-requirements.txt
```

### 2. Start Backend (FastAPI).
```bash
cd backend
uvicorn app.main:app --reload
```

### 3. Start Frontend (Streamlit).
```bash
cd frontend
streamlit run app.py
```

### 3. Folder Structure.
```bash
rag-chatbot/
│
├── backend/
│   ├── app/
│   │   ├── api/                # FastAPI routes
│   │   ├── services/           # Embedding, OCR, Chunking, Retrieval
│   │   └── main.py
│   ├── data/                   # Uploaded and processed documents.
│   │   ├── chroma/ 
│   │   ├── extracted/
│   │   ├── uploads/
│   └── .env
├── streamlit_app/              # Streamlit UI.
│   └── app/
├── documents/                  # Documentation.
├── requirements.txt
├── frontend-requirements.txt
├── render.yaml                 # Render File.
└── README.md                   # This File.
```


📌 Future Improvements

User authentication

Chat history and memory

Live web scraping + answering

Advanced citation formatting

Multi-language support

🧑‍💻 Author
Aaditya Sharma

📄 License
This project is licensed under the MIT License.