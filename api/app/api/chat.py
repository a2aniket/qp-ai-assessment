from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from app.db.database import get_db
from app.utility.chatwithdoc import generate_response
from app.schemas.chat import ChatInput, ChatResponse
from app.core.constants import CHAT_RESPONSE_MSG, CHAT_ERROR
from app.utility.logging_config import app_logger

router = APIRouter()

@router.post("/chat", response_model=ChatResponse)
async def chat_with_document(chat_input: ChatInput, db: Session = Depends(get_db)):
    """
    Generate a response to a user's question based on the uploaded documents.
    """
    try:
        answer = generate_response(chat_input.question, db)
        app_logger.info(f"[Chat] {CHAT_RESPONSE_MSG}")
        return ChatResponse(answer=answer)
    except Exception as e:
        app_logger.error(f"[Chat] {CHAT_ERROR}: {str(e)}")
        raise HTTPException(status_code=500, detail=CHAT_ERROR)
