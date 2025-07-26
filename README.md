# HackRx: LLM-Powered Intelligent Query–Retrieval System

## Overview

This project is a complete and robust solution for the LLM-Powered Intelligent Query–Retrieval System challenge. It is a FastAPI application that processes large documents (PDFs) from a URL and answers natural language questions based on the document's content.

The system is designed to be accurate, efficient, and explainable, directly addressing all the specified evaluation criteria.

---

## How to Run the Solution

To run this project locally, please follow these steps:

1.  **Create a Virtual Environment:**
    ```bash
    python -m venv venv
    ```

2.  **Activate the Environment:**
    *   On Windows: `.\venv\Scripts\activate`
    *   On macOS/Linux: `source venv/bin/activate`

3.  **Install Dependencies:**
    Use the provided `requirements.txt` file to install all necessary packages.
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set Up Environment Variables:**
    *   Rename the `.env.example` file to `.env`.
    *   Open the `.env` file and enter your personal API keys for OpenAI and Pinecone.

5.  **Run the API Server:**
    ```bash
    uvicorn main:app --reload
    ```
    The server will be running at `http://localhost:8000`.

---

## Testing the API

The API can be tested using the automatic Swagger UI documentation.

1.  Once the server is running, navigate to **`http://localhost:8000/docs`** in your web browser.
2.  Click the "Authorize" button and enter the bearer token: `Bearer 35d0ced95b20ee0f825d7b4e9e7bbc9e6fafbfe5a5d6ded15e30e8355eb321be`.
3.  Expand the `POST /hackrx/run` endpoint and click "Try it out".
4.  Use the sample request body provided in the problem description to execute a request. The system will return a detailed JSON response.

---

## Meeting the Evaluation Criteria

This solution was engineered to excel in all five evaluation categories:

**a) Accuracy:**
The system uses a custom-engineered prompt template (`PROMPT_TEMPLATE` in `core_logic.py`) that strictly instructs the LLM to answer *only* based on the provided document context. This minimizes hallucinations and ensures high-precision clause matching.

**b) Token Efficiency / Cost-Effectiveness:**
The solution defaults to using the `gpt-3.5-turbo` model, which provides an excellent balance of high performance at a much lower cost and token usage compared to GPT-4. This demonstrates a practical, production-oriented mindset.

**c) Latency:**
By using `gpt-3.5-turbo` and the highly efficient Pinecone vector database for retrieval, the system is optimized for low-latency, real-time responses after the initial document processing.

**d) Reusability / Extensibility:**
The code is logically separated into `main.py` (API and web logic) and `core_logic.py` (AI and document processing logic). This modular design makes the system easy to maintain, extend, and reuse. For example, adding a new document type (like DOCX) would only require a change in `core_logic.py` without affecting the API layer.

**e) Explainability / Clause Traceability:**
This is a core feature of the solution. The API response does not just provide an answer; it provides a structured object that includes:
*   The original `question`.
*   The generated `answer`.
*   The exact `source_quote` from the document that was used to generate the answer.
*   The `source_page_number` where the quote can be found.

This provides clear, transparent, and traceable reasoning for every decision the system makes.uvicorn main:app --reload