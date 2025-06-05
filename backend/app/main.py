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
# Replace the embedding function import with:
from app.services.embedding import DeepSeekEmbeddingFunction
import requests as http_requests
from typing import List, Dict



app = FastAPI()

# Configuration
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize ChromaDB
client = chromadb.PersistentClient(path="chroma_db")
deepseek_ef = DeepSeekEmbeddingFunction(api_key="sk-4997e1a23a8f4a74ba35e0c870908193")
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

@app.post("/upload/pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    try:
        # Save the uploaded file
        file_id = str(uuid.uuid4())
        file_path = os.path.join(UPLOAD_FOLDER, f"{file_id}.pdf")
        
        with open(file_path, "wb") as f:
            f.write(await file.read())
        
        # Process PDF
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
        # Save the uploaded file
        file_id = str(uuid.uuid4())
        file_path = os.path.join(UPLOAD_FOLDER, f"{file_id}.mp4")
        
        with open(file_path, "wb") as f:
            f.write(await file.read())
        
        # Extract audio from video
        video = VideoFileClip(file_path)
        audio = video.audio
        
        # Save audio to temporary file
        with NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
            audio_path = temp_audio.name
            audio.write_audiofile(audio_path, codec='pcm_s16le')
        
        # Transcribe audio
        recognizer = sr.Recognizer()
        with sr.AudioFile(audio_path) as source:
            audio_data = recognizer.record(source)
            try:
                text = recognizer.recognize_google(audio_data)  # Free option
            except sr.UnknownValueError:
                text = ""
        
        # Clean up
        os.unlink(audio_path)
        video.close()
        
        if text:
            # Split into chunks (simplified - in production you'd want better segmentation)
            chunks = [text[i:i+1000] for i in range(0, len(text), 1000)]
            
            documents = []
            metadatas = []
            ids = []
            
            for i, chunk in enumerate(chunks):
                doc_id = f"{file_id}_chunk_{i + 1}"
                documents.append(chunk)
                metadatas.append({
                    "source": file.filename,
                    "timestamp": f"{i * 10}:00",  # Simplified timestamp
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
        # Retrieve relevant documents
        results = collection.query(
            query_texts=[query.question],
            n_results=5
        )
        
        # Prepare context for LLM
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
        
        # Get real response from DeepSeek
        answer = get_deepseek_response(query.question, context)
        
        return {
            "answer": answer,
            "sources": sources
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        
def get_deepseek_response(question: str, context: str) -> str:
    """
    Get response from DeepSeek API
    """
    api_url = "https://api.deepseek.com/v1/chat/completions"  # Verify the actual endpoint
    headers = {
        "Authorization": "sk-4997e1a23a8f4a74ba35e0c870908193",
        "Content-Type": "application/json"
    }
    
    prompt = f"""
    You are an AI assistant answering questions based on provided context.
    Always cite your sources using the provided metadata.
    
    Context:
    {context}
    
    Question: {question}
    
    Answer in detail with proper citations:
    """
    
    data = {
        "model": "deepseek-chat",  # Verify the correct model name
        "messages": [
            {"role": "system", "content": "You are a helpful assistant that answers questions based on provided documents."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7,
        "max_tokens": 2000
    }
    
    try:
        response = http_requests.post(api_url, headers=headers, json=data)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"Error calling DeepSeek API: {str(e)}")
        return f"Sorry, I couldn't process your question. Error: {str(e)}"
    

@app.get("/status/")
async def status():
    return {"status": "ok", "timestamp": datetime.now().isoformat()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)