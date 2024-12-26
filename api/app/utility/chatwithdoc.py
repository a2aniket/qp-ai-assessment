from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import FAISS
from app.core.constants import EMBEDDING_MODEL_NAME, EMBEDDING_PATH, MODEL_KWARGS, ENCODE_KWARGS
from app.core.config import settings
from langchain_community.llms import HuggingFaceHub

def generate_response(question, db):
    """
    Generate a response to the user's question based on the document embeddings.
    """
    huggingface_embeddings = HuggingFaceBgeEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs=MODEL_KWARGS,
        encode_kwargs=ENCODE_KWARGS
    )

    vectorstore = FAISS.load_local(EMBEDDING_PATH, huggingface_embeddings,allow_dangerous_deserialization=True)
    retriever = vectorstore.as_retriever(
        search_type=settings.RETRIEVER_SEARCH_TYPE,
        search_kwargs={"k": settings.RETRIEVER_K}
    )
 
    llm = HuggingFaceHub(
        repo_id=settings.HUGGINGFACE_REPO_ID,
        model_kwargs={
            "temperature": settings.HUGGINGFACE_TEMPERATURE,
            "max_length": settings.HUGGINGFACE_MAX_LENGTH
        }
    )

    prompt_template = """
        Use the following piece of context to answer the question asked.
        Please try to provide the answer only based on the context

        {context}
        Question:{question}

        Helpful Answers:       
        """
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    retrievalQA = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type=settings.CHAIN_TYPE,
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )
    
    result = retrievalQA({"query": question})

    return result["result"]

    return result["result"]  # Assuming the answer is in the "result" key