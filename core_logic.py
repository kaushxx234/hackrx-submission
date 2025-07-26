# core_logic.py (Final Serverless-Ready Version)

import os
import requests
import io  # ADD THIS NEW IMPORT for in-memory operations
import pypdf # ADD THIS NEW IMPORT for direct PDF processing

from pinecone import Pinecone as PineconeClient, ServerlessSpec
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document # ADD THIS NEW IMPORT to manually create documents
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

PROMPT_TEMPLATE = """
You are a highly intelligent and precise AI assistant for analyzing insurance policy documents.
Use the following document excerpts (CONTEXT) to answer the user's QUESTION.
Your answer must be based *only* on the provided context.
If the context does not contain the answer, you must state: "The answer is not available in the provided document."
Do not make up answers. Be concise and clear.

CONTEXT:
{context}

QUESTION:
{question}

YOUR ANSWER:
"""

# --- REWRITTEN FUNCTION TO WORK IN MEMORY ---
def load_and_split_document(url: str):
    """
    Downloads a PDF from a URL and processes it entirely in memory,
    making it suitable for read-only serverless environments.
    """
    print("Downloading PDF from URL...")
    response = requests.get(url)
    response.raise_for_status()

    # Wrap the downloaded PDF content in an in-memory binary stream
    pdf_file = io.BytesIO(response.content)
    
    # Use pypdf to read the in-memory PDF
    pdf_reader = pypdf.PdfReader(pdf_file)
    
    # Manually create LangChain Document objects for each page
    # This preserves metadata like page numbers.
    documents = []
    for i, page in enumerate(pdf_reader.pages):
        page_content = page.extract_text()
        if page_content:
            documents.append(Document(
                page_content=page_content,
                metadata={"page": i + 1, "source": url}
            ))
            
    # Split the documents into smaller chunks for processing
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    print("Splitting documents...")
    return text_splitter.split_documents(documents)


def create_qa_chain(api_keys: dict):
    """Initializes all components needed for the QA chain."""
    embeddings = OpenAIEmbeddings(openai_api_key=api_keys["openai"])
    pc = PineconeClient(api_key=api_keys["pinecone_key"])
    index_name = "hackrx-index"

    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=1536,
            metric='cosine',
            spec=ServerlessSpec(cloud='aws', region='us-east-1')
        )

    def build_chain(documents):
        vector_store = PineconeVectorStore.from_documents(
            documents, embeddings, index_name=index_name
        )
        retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})
        
        llm = ChatOpenAI(
            temperature=0.0,
            model_name='gpt-3.5-turbo',
            openai_api_key=api_keys["openai"]
        )
        
        prompt = PromptTemplate(template=PROMPT_TEMPLATE, input_variables=["context", "question"])
        chain_type_kwargs = {"prompt": prompt}
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs=chain_type_kwargs,
            return_source_documents=True
        )
        return qa_chain

    return build_chain
