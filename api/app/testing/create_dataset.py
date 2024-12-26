import PyPDF2
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List
import os
from langsmith import Client
from datetime import datetime
from run_test import run_all_evaluations

os.environ["LANGCHAIN_TRACING_V2"]=""
os.environ["LANGCHAIN_ENDPOINT"]=""
os.environ["LANGCHAIN_API_KEY"]=""
os.environ["LANGCHAIN_PROJECT"]=""

llm = ChatGoogleGenerativeAI(model="gemini-pro")

class QAPair(BaseModel):
    inputs: List[str] = Field(description="List of 5 questions")
    outputs: List[str] = Field(description="List of 5 corresponding answers")

output_parser = PydanticOutputParser(pydantic_object=QAPair)

prompt_template = ChatPromptTemplate.from_template(
    """
    Given the following document, generate 5 question-answer pairs about the content.
    
    Document:
    {document}
    
    {format_instructions}
    """
)

def create_langsmith_dataset(pdf_path):
    try:
        document_text = extract_text_from_pdf(pdf_path)
        prompt = prompt_template.format(
            document=document_text[:1000],
            format_instructions=output_parser.get_format_instructions()
        )
        response = llm.invoke(prompt)
        parsed_output = output_parser.parse(response.content)

        client = Client()
        file_name = os.path.basename(pdf_path).split('.')[0]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dataset_name = f"{file_name}_{timestamp}"

        dataset = client.create_dataset(
            dataset_name=dataset_name,
            description=f"QA pairs generated from {file_name}",
        )
        print(f"Dataset created with ID: {dataset.id}")

        inputs = [{"question": q} for q in parsed_output.inputs]
        outputs = [{"answer": a} for a in parsed_output.outputs]

        client.create_examples(
            inputs=inputs,
            outputs=outputs,
            dataset_id=dataset.id,
        )
        print(f"Added {len(inputs)} examples to the dataset")

        return dataset_name
    except Exception as e:
        print(f"Error creating dataset: {str(e)}")
        raise

def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
    return text


if __name__ == "__main__":
    pdf_path = "C:/code/qp-ai-assessment/api/misc/fileupload/falcon-users-guide-2021-09.pdf"  # Replace with your PDF path
    dataset_name = create_langsmith_dataset(pdf_path)
    print(f"Dataset created: {dataset_name}")
    
    # Run all evaluations on the newly created dataset
    evaluation_results = run_all_evaluations(dataset_name)
    print("Evaluation results:")
    for evaluation_type, result in evaluation_results.items():
        print(f"{evaluation_type}:")
        print(result)
        print("---")
    



