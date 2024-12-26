from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from app.core.config import settings

def create_embedding(file):
    """
    Create embedding for the given file.
    """
    loader = PyPDFLoader(file)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    final_documents = text_splitter.split_documents(documents)

    huggingface_embeddings = HuggingFaceBgeEmbeddings(
        model_name=settings.EMBEDDING_MODEL_NAME, 
        model_kwargs=settings.MODEL_KWARGS,
        encode_kwargs=settings.ENCODE_KWARGS
    )
    vectorstore = FAISS.from_documents(final_documents, huggingface_embeddings)
    vectorstore.save_local(settings.EMBEDDING_PATH)
    
    return "embedding_created_successfully"
