from pydantic import BaseModel

class ChatInput(BaseModel):
    question: str

class ChatResponse(BaseModel):
    answer: str
