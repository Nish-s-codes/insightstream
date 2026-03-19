from fastapi import APIRouter, UploadFile
import os
from app.services.rag import answer_question
from app.core.pipeline import process_document

router = APIRouter()

UPLOAD_DIR = "data/uploads"

@router.post("/upload")
async def upload_file(file: UploadFile):
    file_path = os.path.join(UPLOAD_DIR, file.filename)

    with open(file_path, "wb") as f:
        f.write(await file.read())

    result = process_document(file_path)

    return {
        "message": "Processed",
        "result": result
    }
    

@router.get("/query")
def query(q: str):
    return answer_question(q)