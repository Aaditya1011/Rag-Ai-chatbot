from fastapi import FastAPI
from app.api.routes import router

# FastApi app initialisation.
app = FastAPI(title="RAG AI Chatbot")
app.include_router(router)

# home/root route.
@app.get('/')
def root():
    return {"message":"RAG chatbot backend is running."}
