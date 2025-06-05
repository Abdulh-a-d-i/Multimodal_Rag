import os
import uuid
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional
import chromadb
from chromadb.utils import embedding_functions
import PyPDF2
from moviepy.editor import VideoFileClip
import speech_recognition as sr
from tempfile import NamedTemporaryFile
from datetime import datetime
import requests

app = FastAPI()

# Configuration
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize ChromaDB with DeepSeek embeddings
client = chromadb.PersistentClient(path="chroma_db")
deepseek_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="deepseek-ai/deepseek-embedding")
collection = client.get_or_create_collection(
    name="multimodal_rag",
    embedding_function=deepseek_ef
)

class Query(BaseModel):
    question: str

class DocumentResponse(BaseModel):
    text: str
    metadata: dict

class VideoResponse(BaseModel):
    text: str
    metadata: dict

# DeepSeek API Config
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")  # Set your API key in environment variables
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"

@app.post("/upload/pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    try:
        file_id = str(uuid.uuid4())
        file_path = os.path.join(UPLOAD_FOLDER, f"{file_id}.pdf")
        
        with open(file_path, "wb") as f:
            f.write(await file.read())
        
        with open(file_path, "rb") as f:
            pdf_reader = PyPDF2.PdfReader(f)
            documents = []
            metadatas = []
            ids = []
            
            for page_num, page in enumerate(pdf_reader.pages):
                text = page.extract_text()
                if text.strip():
                    doc_id = f"{file_id}_page_{page_num + 1}"
                    documents.append(text)
                    metadatas.append({
                        "source": file.filename,
                        "page": page_num + 1,
                        "type": "pdf",
                        "file_id": file_id
                    })
                    ids.append(doc_id)
            
            if documents:
                collection.add(
                    documents=documents,
                    metadatas=metadatas,
                    ids=ids
                )
        
        return {"message": "PDF processed successfully", "file_id": file_id}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload/video/")
async def upload_video(file: UploadFile = File(...)):
    try:
        file_id = str(uuid.uuid4())
        file_path = os.path.join(UPLOAD_FOLDER, f"{file_id}.mp4")
        
        with open(file_path, "wb") as f:
            f.write(await file.read())
        
        video = VideoFileClip(file_path)
        audio = video.audio
        
        with NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
            audio_path = temp_audio.name
            audio.write_audiofile(audio_path, codec='pcm_s16le')
        
        recognizer = sr.Recognizer()
        with sr.AudioFile(audio_path) as source:
            audio_data = recognizer.record(source)
            try:
                text = recognizer.recognize_google(audio_data)
            except sr.UnknownValueError:
                text = ""
        
        os.unlink(audio_path)
        video.close()
        
        if text:
            chunks = [text[i:i+1000] for i in range(0, len(text), 1000)]
            documents = []
            metadatas = []
            ids = []
            
            for i, chunk in enumerate(chunks):
                doc_id = f"{file_id}_chunk_{i + 1}"
                documents.append(chunk)
                metadatas.append({
                    "source": file.filename,
                    "timestamp": f"{i * 10}:00",
                    "type": "video",
                    "file_id": file_id
                })
                ids.append(doc_id)
            
            collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
        
        return {"message": "Video processed successfully", "file_id": file_id}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query/")
async def query_rag(query: Query):
    try:
        results = collection.query(
            query_texts=[query.question],
            n_results=5
        )
        
        context = ""
        sources = []
        
        for doc, metadata in zip(results['documents'][0], results['metadatas'][0]):
            context += f"Source ({metadata['type']} from {metadata['source']}): {doc}\n\n"
            
            if metadata['type'] == "pdf":
                sources.append({
                    "type": "pdf",
                    "source": metadata['source'],
                    "page": metadata['page'],
                    "file_id": metadata['file_id'],
                    "content": doc[:200] + "..."
                })
            else:
                sources.append({
                    "type": "video",
                    "source": metadata['source'],
                    "timestamp": metadata['timestamp'],
                    "file_id": metadata['file_id'],
                    "content": doc[:200] + "..."
                })
        
        answer = get_deepseek_response(query.question, context)
        
        return {
            "answer": answer,
            "sources": sources
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def get_deepseek_response(question: str, context: str) -> str:
    """Get response directly from DeepSeek API"""
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }
    
    prompt = f"""
    Answer based on this context (cite sources exactly as shown):
    
    {context}
    
    Question: {question}
    """
    
    data = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant that provides detailed answers with citations."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7,
        "max_tokens": 2000
    }
    
    try:
        response = requests.post(DEEPSEEK_API_URL, headers=headers, json=data)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"Error: {str(e)}"

@app.get("/status/")
async def status():
    return {"status": "ok", "timestamp": datetime.now().isoformat()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
