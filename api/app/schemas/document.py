from pydantic import BaseModel

class DocumentBase(BaseModel):
    filename: str

class DocumentCreate(DocumentBase):
    pass

class Document(DocumentBase):
    id: int
    embedding: str

    class Config:
        from_attributes = True
