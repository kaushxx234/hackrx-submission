# core_logic.py (Winning Version)

import os
import requests
from pinecone import Pinecone as PineconeClient, ServerlessSpec
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# --- 1. Enhanced Explainability and Accuracy with a Custom Prompt ---
# This prompt template instructs the AI on how to behave, improving accuracy and ensuring
# it uses the provided documents effectively.
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

# --- Function to Load and Process the Document ---
def load_and_split_document(url: str):
    """Downloads a PDF, loads it, and splits it into chunks."""
    try:
        response = requests.get(url)
        response.raise_for_status()
        with open("temp_document.pdf", "wb") as f:
            f.write(response.content)
        loader = PyPDFLoader("temp_document.pdf")
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        return text_splitter.split_documents(documents)
    finally:
        if os.path.exists("temp_document.pdf"):
            os.remove("temp_document.pdf")

# --- Function to Create the Question-Answering Chain ---
def create_qa_chain(api_keys: dict):
    """Initializes all components needed for the QA chain."""
    
    # --- 2. Optimized for Token Efficiency and Latency ---
    # We use OpenAIEmbeddings for efficient document embedding.
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
        
        # Using a more cost-effective and faster model for better Latency and Token Efficiency.
        llm = ChatOpenAI(
            temperature=0.0,
            model_name='gpt-3.5-turbo', # Cheaper and faster than GPT-4
            openai_api_key=api_keys["openai"]
        )
        
        # Integrating our custom prompt into the chain
        prompt = PromptTemplate(template=PROMPT_TEMPLATE, input_variables=["context", "question"])
        chain_type_kwargs = {"prompt": prompt}
        
        # --- 3. Ensures we get source documents for Explainability ---
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs=chain_type_kwargs,
            return_source_documents=True # This is crucial!
        )
        return qa_chain

    return build_chain