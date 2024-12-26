# Document Chat API

This project provides an API for uploading documents, creating embeddings, and chatting with the documents.

## Setup

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run the application: `uvicorn app.main:app --reload`

## Docker

To run the application using Docker:

1. Build the Docker image: `docker build -t doc-chat-api .`
2. Run the container: `docker run -p 8000:8000 doc-chat-api`

## API Documentation

Once the application is running, you can access the Swagger UI at `http://localhost:8000/docs`.

## Endpoints

- POST /api/v1/upload: Upload a document (PDF or Word) and create its embedding
- POST /api/v1/chat: Chat with the uploaded documents

For detailed information about request/response formats, please refer to the Swagger documentation.
