import os
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from sqlalchemy.orm import Session
from app.db.database import get_db
from app.utility.embedding import create_embedding
from app.schemas.document import Document
from app.models.document import Document as DocumentModel
from app.core.constants import EMBEDDING_SUCCESS_MSG, INVALID_FILE_TYPE_ERROR, EMBEDDING_ERROR,FILE_UPLOAD_DIR
from app.utility.logging_config import app_logger
from langsmith import traceable

router = APIRouter()


@router.post("/upload", response_model=Document)
@traceable
async def upload_document(file: UploadFile = File(...), db: Session = Depends(get_db)):
    """
    Upload a document (PDF or Word), save it to misc/fileupload/, and create its embedding.
    """
    if not file.filename.lower().endswith(('.pdf', '.docx')):
        app_logger.error(f"[{file.filename}] {INVALID_FILE_TYPE_ERROR}")
        raise HTTPException(status_code=400, detail=INVALID_FILE_TYPE_ERROR)

    try:
        # Create the directory if it doesn't exist
        upload_dir = FILE_UPLOAD_DIR
        os.makedirs(upload_dir, exist_ok=True)

        # Save the file
        file_path = os.path.join(upload_dir, file.filename)
        with open(file_path, "wb") as buffer:
            buffer.write(await file.read())

        embedding = create_embedding(file_path)
        
        db_document = DocumentModel(filename=file.filename, embedding=embedding, file_path=file_path)
        db.add(db_document)
        db.commit()
        db.refresh(db_document)
        app_logger.info(f"[{file.filename}] {EMBEDDING_SUCCESS_MSG}")
        return db_document
    except Exception as e:
        app_logger.error(f"[{file.filename}] {EMBEDDING_ERROR}: {str(e)}")
        raise HTTPException(status_code=500, detail=EMBEDDING_ERROR)
