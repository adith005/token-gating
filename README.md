# Token Optimiser v1.5

A multi-source context engine integrating Redis memory (MCP), RAG, and Token Gating to optimize token usage in LLM prompts.

## Introduction

This project provides a system for intelligently managing the context provided to Large Language Models (LLMs). It reduces the number of tokens sent in a prompt by selecting only the most relevant information from various sources, such as conversational history and external documents. This leads to faster response times and lower operational costs, while preserving the quality of the generated output.

## Features

-   **Hybrid Context:** Combines conversational history (from Redis) and external documents (RAG).
-   **Token Gating:** Smartly selects the most relevant context to fit within a specified token budget.
-   **Multiple Interfaces:** Interact with the system via a command-line interface (CLI), a REST API, or a Streamlit-based web UI.
-   **Extensible:** Designed to be modular, allowing for the easy addition of new data sources or custom gating strategies.
-   **Containerized:** Comes with a Dockerfile for easy setup and deployment.

## Prerequisites

Before you begin, ensure you have the following installed:
-   Python 3.10+
-   Docker

## Setup Instructions

1.  **Clone the Repository**
    ```bash
    git clone <your-repository-url>
    cd tokenoptimizer_org
    ```

2.  **Create and Activate a Virtual Environment**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install Dependencies**
    The `requirements.txt` file is required to install the necessary packages.
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set Up Environment Variables**
    Create a `.env` file by copying the example file:
    ```bash
    cp .env.example .env
    ```
    Now, edit the `.env` file and add your credentials and configurations:
    -   `OPENAI_API_KEY`: Your API key for OpenAI services.
    -   `CHROMA_DIR`: Directory to store ChromaDB persistence files. (e.g., `chroma_db/`)
    -   `FAISS_INDEX_PATH`: Path to save the FAISS index file. (e.g., `faiss_index.bin`)

5.  **Start Redis Instance**
    Use Docker to run a Redis instance, which is used for memory and caching.
    ```bash
    docker run -d --name redis -p 6379:6379 redis:latest
    ```

## Data Ingestion

You can load data into the system in two ways:

### Ingesting PDF Documents
To process and embed PDF files for the RAG system:
```bash
python -m rag.document_ingestor path/to/your.pdf
```
This will chunk the document, generate embeddings, and store them.

### Loading from JSONL
You can also load data from a JSONL file. A sample file is provided at `data/sample_memory.jsonl`.
```bash
python scripts/embed_index.py data/sample_memory.jsonl
```

## Running the Application

This project can be run in several modes:

### 1. CLI Mode
For an interactive command-line session:
```bash
python main.py
```

### 2. API Server
To start the FastAPI server:
```bash
uvicorn app.server:app --reload --port 8000
```
You can then send requests to the API. For example, using `curl`:
```bash
curl -X POST "http://localhost:8000/gate" \
     -H "Content-Type: application/json" \
     -d 
           "query": "What is token optimization?",
           "strategy": "heuristic",
           "top_k": 5,
           "token_budget": 512
         }"
```

### 3. Web Frontend
To run the Streamlit web interface:
```bash
streamlit run Frontend/frontend.py
```
Navigate to the URL provided by Streamlit in your browser.

## Running Evaluation

To evaluate the system's performance with a specific query:
```bash
python main.py --evaluate "your query here"
```

## Running with Docker

You can also build and run the entire application as a Docker container.

1.  **Build the Docker Image**
    ```bash
    docker build -t token-optimiser .
    ```

2.  **Run the Container**
    This command runs the application in interactive mode, mounts the current directory, and passes the environment variables from your `.env` file.
    ```bash
    docker run -it --rm \
      -v $(pwd):/app \
      --env-file .env \
      --network="host" \
      token-optimiser
    ```
    *Note: `--network="host"` is used to allow the application inside the container to connect to the Redis instance running on `localhost`.*

## Project Structure

-   `app/`: Contains the FastAPI server code.
-   `gating/`: Core logic for the token gating mechanism.
-   `llm/`: Interface for interacting with the Language Model (e.g., OpenAI).
-   `memory/`: Manages conversational history and memory using Redis.
-   `rag/`: Handles the Retrieval-Augmented Generation, including document ingestion and retrieval.
-   `scripts/`: Utility scripts for tasks like data loading.
-   `Frontend/`: Contains the Streamlit frontend application.
-   `tests/`: Unit and integration tests.
-   `main.py`: Main entry point for the CLI application.
-   `Dockerfile`: For building the Docker container.
-   `.env.example`: Example file for environment variables.
-   `requirements.txt`: Python dependencies.