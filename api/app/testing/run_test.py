
import os
from typing import List
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFaceHub
from langchain.schema import Document
from langsmith import traceable
from langsmith.evaluation import LangChainStringEvaluator, evaluate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

# Environment setup
def setup_environment():
    os.environ["LANGCHAIN_TRACING_V2"] = ""
    os.environ["LANGCHAIN_ENDPOINT"] = ""
    os.environ["LANGCHAIN_API_KEY"] = ""
    os.environ["LANGCHAIN_PROJECT"] = ""


EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
EMBEDDING_PATH = "C:/code/qp-ai-assessment/api/misc/embeddings/"
MODEL_KWARGS = {"device": "cpu"}
ENCODE_KWARGS = {"normalize_embeddings": True}

# Initialize LLM and vectorstore
def initialize_llm_and_vectorstore():
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-pro",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
    )
    
    huggingface_embeddings = HuggingFaceBgeEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs=MODEL_KWARGS,
        encode_kwargs=ENCODE_KWARGS
    )
    
    vectorstore = FAISS.load_local(EMBEDDING_PATH, huggingface_embeddings, allow_dangerous_deserialization=True)
    retriever = vectorstore.as_retriever()
    return llm, retriever

# RagBot class
class RagBot:
    def __init__(self, retriever, repo_id: str = "mistralai/Mistral-7B-v0.1"):
        self._retriever = retriever
        self._llm = HuggingFaceHub(
            repo_id=repo_id,
            model_kwargs={"temperature": 0.1, "max_length": 500}
        )

    @traceable()
    def retrieve_docs(self, question):
        return self._retriever.get_relevant_documents(question)

    @traceable()
    def invoke_llm(self, question: str, docs: List[Document]):
        docs_content = "\n\n".join([doc.page_content for doc in docs])
        system_prompt = (
            "You are a helpful AI code assistant with expertise in LCEL. "
            "Use the following docs to produce a concise code solution to the user question.\n\n"
            f"## Docs\n\n{docs_content}\n\n"
        )
        
        input_text = f"{system_prompt}Human: {question}\nAssistant:"
        
        response = self._llm(input_text)
        
        return {
            "answer": response,
            "contexts": [doc.page_content for doc in docs],
        }

    @traceable()
    def get_answer(self, question: str):
        docs = self.retrieve_docs(question)
        return self.invoke_llm(question, docs)

# Prediction functions
def predict_rag_answer(example: dict):
    response = rag_bot.get_answer(example["question"])
    return {"answer": response["answer"]}

def predict_rag_answer_with_context(example: dict):
    response = rag_bot.get_answer(example["question"])
    return {"answer": response["answer"], "contexts": response["contexts"]}

# Evaluation functions
def run_qa_evaluation(dataset_name):
    qa_evaluator = [
        LangChainStringEvaluator(
            "cot_qa",
            config={"llm": llm},
            prepare_data=lambda run, example: {
                "prediction": run.outputs["answer"],
                "reference": example.outputs["answer"],
                "input": example.inputs["question"],
            },
        )
    ]

    return evaluate(
        predict_rag_answer,
        data=dataset_name,
        evaluators=qa_evaluator,
        experiment_prefix="rag-qa-evaluation",
        metadata={"variant": "LCEL context, Mistral-7B"},
    )

def run_hallucination_evaluation(dataset_name):
    answer_hallucination_evaluator = LangChainStringEvaluator(
        "labeled_score_string",
        config={
            "criteria": {
                "accuracy": """Is the Assistant's Answer grounded in the Ground Truth documentation? A score of [[1]] means that the
                Assistant answer contains is not at all based upon / grounded in the Ground Truth documentation. A score of [[5]] means 
                that the Assistant answer contains some information (e.g., a hallucination) that is not captured in the Ground Truth 
                documentation. A score of [[10]] means that the Assistant answer is fully based upon the in the Ground Truth documentation."""
            },
            "llm": llm,
            "normalize_by": 10,
        },
        prepare_data=lambda run, example: {
            "prediction": run.outputs["answer"],
            "reference": run.outputs["contexts"],
            "input": example.inputs["question"],
        },
    )

    return evaluate(
        predict_rag_answer_with_context,
        data=dataset_name,
        evaluators=[answer_hallucination_evaluator],
        experiment_prefix="hallucination",
        metadata={"variant": "LCEL context, gpt-3.5-turbo"},
    )

def run_docs_relevance_evaluation(dataset_name):
    docs_relevance_evaluator = LangChainStringEvaluator(
        "score_string",
        config={
            "criteria": {
                "document_relevance": """The response is a set of documents retrieved from a vectorstore. The input is a question
                used for retrieval. You will score whether the Assistant's response (retrieved docs) is relevant to the Ground Truth 
                question. A score of [[1]] means that none of the  Assistant's response documents contain information useful in answering or addressing the user's input.
                A score of [[5]] means that the Assistant answer contains some relevant documents that can at least partially answer the user's question or input. 
                A score of [[10]] means that the user input can be fully answered using the content in the first retrieved doc(s)."""
            },
            "llm": llm,
            "normalize_by": 10,
        },
        prepare_data=lambda run, example: {
            "prediction": run.outputs["contexts"],
            "input": example.inputs["question"],
        },
    )

    return evaluate(
        predict_rag_answer_with_context,
        data=dataset_name,
        evaluators=[docs_relevance_evaluator],
        experiment_prefix="doc-relevance",
        metadata={"variant": "LCEL context, gpt-3.5-turbo"},
    )

def run_custom_evaluations(dataset_name):
    return evaluate(
        predict_rag_answer,
        data=dataset_name,
        evaluators=[document_relevance_grader, answer_hallucination_grader],
        experiment_prefix="LCEL context"
    )

# Custom evaluation functions
def document_relevance_grader(root_run, example):
    rag_pipeline_run = next(run for run in root_run.child_runs if run.name == "get_answer")
    retrieve_run = next(run for run in rag_pipeline_run.child_runs if run.name == "retrieve_docs")
    doc_txt = "\n\n".join(doc.page_content for doc in retrieve_run.outputs["output"])
    question = retrieve_run.inputs["question"] 

    class GradeDocuments(BaseModel):
        binary_score: int = Field(description="Documents are relevant to the question, 1 or 0")
    
    structured_llm_grader = llm.with_structured_output(GradeDocuments)
    
    system = """You are a grader assessing relevance of a retrieved document to a user question. \n 
        If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
        It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
        Give a binary score 1 or 0 score, where 1 means that the document is relevant to the question."""
    
    grade_prompt = ChatPromptTemplate.from_messages([
        ("system", system),
        ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
    ])
    
    retrieval_grader = grade_prompt | structured_llm_grader
    score = retrieval_grader.invoke({"question": question, "document": doc_txt})
    return {"key": "document_relevance", "score": int(score.binary_score)}

def answer_hallucination_grader(root_run, example):
    rag_pipeline_run = next(run for run in root_run.child_runs if run.name == "get_answer")
    retrieve_run = next(run for run in rag_pipeline_run.child_runs if run.name == "retrieve_docs")
    doc_txt = "\n\n".join(doc.page_content for doc in retrieve_run.outputs["output"])
    generation = rag_pipeline_run.outputs["answer"]
    
    class GradeHallucinations(BaseModel):
        binary_score: int = Field(description="Answer is grounded in the facts, 1 or 0")
    
    structured_llm_grader = llm.with_structured_output(GradeHallucinations)
    
    system = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. \n 
         Give a binary score 1 or 0, where 1 means that the answer is grounded in / supported by the set of facts."""
    hallucination_prompt = ChatPromptTemplate.from_messages([
        ("system", system),
        ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"),
    ])
    
    hallucination_grader = hallucination_prompt | structured_llm_grader
    score = hallucination_grader.invoke({"documents": doc_txt, "generation": generation})
    return {"key": "answer_hallucination", "score": int(score.binary_score)}

# Main function to run all evaluations
def run_all_evaluations(dataset_name):
    results = {
        "qa_evaluation": run_qa_evaluation(dataset_name),
        "hallucination_evaluation": run_hallucination_evaluation(dataset_name),
        "docs_relevance_evaluation": run_docs_relevance_evaluation(dataset_name),
        "custom_evaluations": run_custom_evaluations(dataset_name)
    }
    return results

# Initialize global variables
setup_environment()
llm, retriever = initialize_llm_and_vectorstore()
rag_bot = RagBot(retriever)

if __name__ == "__main__":
    # This block will not be executed when imported as a module
    test_dataset_name = "RAG_test_LCEL22"
    results = run_all_evaluations(test_dataset_name)
    print("Evaluation results:", results)
