# Chat with Document

Chat with Doc: Converse with your documents. Ask questions, get insights from your PDFs and Word files.

You can access the application at:
http://34.132.23.49:8501/

## Testing and Monitoring

You can view the test reports and monitoring data for this project at:
[qp-ai-assessment/reports](https://github.com/a2aniket/qp-ai-assessment/tree/main/reports)

## Architecture

![chat doc drawio (1)](https://github.com/user-attachments/assets/a0a36531-ea7a-44f3-9fb0-5876a228f417)

## Setup

1. Clone the repository

2. Set up the following environment variables:
   - HUGGINGFACEHUB_API_TOKEN
   - LANGCHAIN_TRACING_V2
   - LANGCHAIN_ENDPOINT
   - LANGCHAIN_API_KEY
   - LANGCHAIN_PROJECT

   You can set these variables in a `.env` file in the root directory of the project or export them in your shell environment.

3. Install dependencies: `pip install -r requirements.txt`

4. Run the API application: `uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload`

## Running the UI

To run the user interface:

1. Navigate to the `ui` folder: `cd ui`

2. Run the Streamlit app: `streamlit run client.py`

3. Open your web browser and go to the URL provided by Streamlit (usually `http://localhost:8501`)

## API Documentation

Once the application is running, you can access the Swagger UI at `http://localhost:8000/docs`.

## Endpoints

- POST /api/v1/upload: Upload a document (PDF or Word) and create its embedding

- POST /api/v1/chat: Chat with the uploaded documents

For detailed information about request/response formats, please refer to the Swagger documentation.
