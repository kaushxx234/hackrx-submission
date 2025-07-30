# core_logic.py (Gemini-Powered Version)

import os
import requests
import io
import pypdf

from pinecone import Pinecone as PineconeClient, ServerlessSpec
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

# --- THE KEY CHANGES ARE HERE ---
# Instead of importing from langchain_openai, we import from langchain_google_genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
# --------------------------------

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

def load_and_split_document(url: str):
    """
    Downloads a PDF from a URL and processes it entirely in memory.
    """
    response = requests.get(url)
    response.raise_for_status()
    pdf_file = io.BytesIO(response.content)
    pdf_reader = pypdf.PdfReader(pdf_file)
    documents = []
    for i, page in enumerate(pdf_reader.pages):
        page_content = page.extract_text()
        if page_content:
            documents.append(Document(
                page_content=page_content,
                metadata={"page": i + 1, "source": url}
            ))
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return text_splitter.split_documents(documents)

def create_qa_chain(api_keys: dict):
    """Initializes all components needed for the QA chain using Gemini."""
    
    # --- SWAP OUT THE EMBEDDINGS MODEL ---
    # Instead of OpenAIEmbeddings, we use Google's model.
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_keys["google"])
    # ---------------------------------------

    pc = PineconeClient(api_key=api_keys["pinecone_key"])
    index_name = "hackrx-index"

    if index_name not in pc.list_indexes().names():
        # Pinecone's free tier only allows one index. If you have an old one,
        # you might need to delete it from their website first.
        pc.create_index(
            name=index_name,
            dimension=768, # IMPORTANT: Google's embedding model has a different dimension size (768)
            metric='cosine',
            spec=ServerlessSpec(cloud='aws', region='us-east-1')
        )

    def build_chain(documents):
        vector_store = PineconeVectorStore.from_documents(
            documents, embeddings, index_name=index_name
        )
        retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})
        
        # --- SWAP OUT THE LANGUAGE MODEL ---
        # Instead of ChatOpenAI, we use ChatGoogleGenerativeAI with a Gemini model.
        llm = ChatGoogleGenerativeAI(
            model="gemini-pro",
            google_api_key=api_keys["google"],
            temperature=0.0,
            convert_system_message_to_human=True # Helps with compatibility
        )
        # -----------------------------------
        
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
