import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    PROJECT_NAME: str = "Document Chat API"
    PROJECT_VERSION: str = "1.0.0"
    ALLOWED_HOSTS: list = ["*"]
    DATABASE_URL: str = f"sqlite:///{os.path.abspath('sql_app.db')}"

    # HuggingFace related settings
    HUGGINGFACE_REPO_ID: str = "mistralai/Mistral-7B-v0.1"
    HUGGINGFACE_TEMPERATURE: float = 0.1
    HUGGINGFACE_MAX_LENGTH: int = 500

    # Embedding related settings (moved from constants.py)
    EMBEDDING_MODEL_NAME: str = "BAAI/bge-small-en-v1.5"
    EMBEDDING_PATH: str = "misc/embeddings/"
    MODEL_KWARGS: dict = {'device': 'cpu'}
    ENCODE_KWARGS: dict = {'normalize_embeddings': True}

    # Retriever and chain settings
    RETRIEVER_SEARCH_TYPE: str = "similarity"
    RETRIEVER_K: int = 3
    CHAIN_TYPE: str = "stuff"

    class Config:
        env_file = ".env"

settings = Settings()
