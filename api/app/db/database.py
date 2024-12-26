import os
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from app.core.config import settings
from app.core.constants import SQLITE_URL_PREFIX
import logging

SQLALCHEMY_DATABASE_URL = settings.DATABASE_URL

# Ensure the directory for the database file exists
if SQLALCHEMY_DATABASE_URL.startswith(SQLITE_URL_PREFIX):
    db_path = SQLALCHEMY_DATABASE_URL[len(SQLITE_URL_PREFIX):]
    os.makedirs(os.path.dirname(db_path), exist_ok=True)

try:
    engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base = declarative_base()
except Exception as e:
    logging.error(f"Failed to create database engine: {str(e)}")
    raise

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()