# main.py (Corrected Indentation)

import os
from fastapi import FastAPI, HTTPException, Security
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
from typing import List, Union
from dotenv import load_dotenv

# This is the new import you added
from fastapi.responses import HTMLResponse 

from core_logic import load_and_split_document, create_qa_chain

load_dotenv()

# --- Authentication Setup ---
API_KEY = "35d0ced95b20ee0f825d7b4e9e7bbc9e6fafbfe5a5d6ded15e30e8355eb321be"
api_key_header = APIKeyHeader(name="Authorization", auto_error=False)

def get_api_key(key: str = Security(api_key_header)):
    if key == f"Bearer {API_KEY}":
        return key
    else:
        raise HTTPException(status_code=403, detail="Could not validate credentials")

# --- Pydantic Models for a Structured, Explainable Response ---
class Answer(BaseModel):
    question: str
    answer: str
    source_quote: str
    source_page_number: Union[int, str]

class QueryRequest(BaseModel):
    documents: str
    questions: List[str]

class AnswerResponse(BaseModel):
    answers: List[Answer]

# --- FastAPI Application Setup ---
app = FastAPI(
    title="Intelligent Query-Retrieval API",
    description="An API that meets all evaluation criteria for the HackRx challenge.",
    version="2.0.0"
)

def get_api_keys_from_env():
    return {
        "google": os.getenv("GOOGLE_API_KEY"),
        "pinecone_key": os.getenv("PINECONE_API_KEY"),
        "pinecone_env": os.getenv("PINECONE_ENVIRONMENT"),
    }

# --- API Endpoint with Authentication and Enhanced Logic ---
@app.post(
    "/hackrx/run", 
    response_model=AnswerResponse, 
    tags=["Query System"],
    dependencies=[Security(get_api_key)]
)
async def run_submission(request: QueryRequest):
    """
    Processes a document and answers questions with full traceability.
    """
    api_keys = get_api_keys_from_env()
    if not all(api_keys.values()):
        raise HTTPException(status_code=500, detail="API keys are not configured correctly on the server.")
    
    try:
        split_docs = load_and_split_document(request.documents)
        if not split_docs:
            raise HTTPException(status_code=500, detail="Failed to process the document.")

        chain_builder = create_qa_chain(api_keys)
        qa_chain = chain_builder(split_docs)

        answers = []
        for question in request.questions:
            try:
                result = qa_chain.invoke({"query": question})
                
                if result['source_documents']:
                    source_doc = result['source_documents'][0]
                    quote = source_doc.page_content
                    page_num = source_doc.metadata.get('page', 'N/A')
                else:
                    quote = "No source document found."
                    page_num = "N/A"

                answers.append({
                    "question": question,
                    "answer": result["result"],
                    "source_quote": quote,
                    "source_page_number": page_num
                })
            except Exception as e:
                answers.append({
                    "question": question, 
                    "answer": f"Error processing this question: {str(e)}", 
                    "source_quote": "", 
                    "source_page_number": "N/A"
                })
        
        return {"answers": answers}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")


# --- This is the function that had the indentation error ---
# It should be at the very edge, with no spaces before it.
@app.get("/", response_class=HTMLResponse, tags=["Root"])
async def read_root():
    """
    Serves the beautiful landing page.
    """
    # This part needs to be indented once.
    with open("index.html") as f:
        return HTMLResponse(content=f.read(), status_code=200)
