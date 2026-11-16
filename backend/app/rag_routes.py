
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel, Field
from google.genai import errors as gemini_errors
from .rag_service import API_KEY

from .rag_service import (
    process_uploaded_document,
    answer_question,
    list_documents,
    delete_document
)
from .rag_service import SHARED_SOURCE

router = APIRouter(
    prefix="/api/rag",
    tags=["RAG Knowledge Assistant"]
)

MAX_FILE_SIZE = 10 * 1024 * 1024

ALLOWED_EXTENSIONS = {
    ".pdf",
    ".docx",
    ".txt"
}


class ChatRequest(BaseModel):
    question: str = Field(
        min_length=2,
        max_length=2000
    )


@router.get("/status")
def status():
    shared_ready = SHARED_SOURCE.exists()
    return {"configured": bool(API_KEY), "guide_available":shared_ready, "message": (
        "The shared guide is available. If Gemini is unavailable, answers switch to labelled PDF passages."
        if shared_ready else "The shared PDF is missing. Ask the administrator to restore it.")}


@router.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    if not API_KEY:
        raise HTTPException(503, "GEMINI_API_KEY is not configured on the backend")
    filename = Path(file.filename or "").name
    extension = Path(filename).suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Upload only PDF, DOCX or TXT files."
        )

    content = await file.read(MAX_FILE_SIZE + 1)

    if not content:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is empty."
        )

    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail="Maximum file size is 10 MB."
        )

    try:
        return process_uploaded_document(
            filename,
            content
        )

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error)
        )

    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Document processing failed. Check backend logs."
        )


@router.post("/chat")
def chat(request: ChatRequest):
    try:
        return answer_question(request.question)
    except gemini_errors.ServerError:
        raise HTTPException(503, "Gemini is temporarily unavailable or timed out. Please try your question again.")
    except Exception:
        raise HTTPException(
            status_code=500,
            detail="AI question answering failed. Check backend logs."
        )


@router.get("/documents")
def documents():
    return {
        "documents": list_documents()
    }


@router.delete("/documents/{document_id}")
def remove_document(document_id: str):
    try:
        deleted = delete_document(document_id)
    except ValueError as exc:
        raise HTTPException(403, str(exc))

    if not deleted:
        raise HTTPException(
            status_code=404,
            detail="Document not found."
        )

    return {
        "message": "Document deleted successfully."
    }
