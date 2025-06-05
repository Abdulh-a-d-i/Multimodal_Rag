# app/api/endpoints/upload.py

from fastapi import APIRouter
from app.services import pdf_processor, video_processor  # Absolute import
from fastapi import UploadFile, File
from fastapi.responses import JSONResponse
router = APIRouter()

@router.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    try:
        # Your processing logic
        return JSONResponse(
            content={"message": "File processed successfully"},
            status_code=200
        )
    except Exception as e:
        return JSONResponse(
            content={"error": str(e)},
            status_code=500
        )
async def upload_file(file: UploadFile = File(...)):
    if file.filename.endswith(".pdf"):
        return await pdf_processor.process_pdf(file)
    elif file.filename.endswith(".mp4"):
        return await video_processor.process_video(file)
    else:
        return {"error": "Unsupported file type."}
