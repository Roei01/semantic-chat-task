# Semantic Legal Chat

A Retrieval Augmented Generation (RAG) system for querying Israeli legal documents (verdicts) with Hebrew support.

## Prerequisites

- Python 3.9+
- Node.js 18+
- Google Chrome (for scraping)
- Ollama (running locally with `llama3` model)
- OpenAI API Key (optional, for cloud model)

## Setup & Installation

1.  **Clone the repository:**

    ```bash
    git clone <repository_url>
    cd <repository_name>
    ```

2.  **Install Python dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

3.  **Install Frontend dependencies:**

    ```bash
    cd frontend
    npm install
    cd ..
    ```

4.  **Set up Ollama:**

    - Download and install [Ollama](https://ollama.com/).
    - Pull the Llama 3 model:
      ```bash
      ollama pull llama3
      ```
    - Ensure Ollama is running (`ollama serve`).

5.  **Set up OpenAI (Optional):**
    - Write your API key:
      create .env file in root of project
      save as: API_GPT="put here the api gpt"

## Usage

### 1. Data Acquisition (Scraping)

Download legal verdicts (Word and PDF) from gov.il:

```bash
python3 scraper/verdict_scraper.py
```

_This will download ~100 documents to `scraper/data`._

### 2. Build Search Index

Create the semantic search index (Vector Store) from the downloaded documents:

```bash
python3 build_index.py
```

_This processes the documents, creates embeddings, and saves them to `vectorstore/`._

### 3. Run the Application

You need to run the backend and frontend in separate terminals.

**Backend (FastAPI):**

```bash
python3 -m uvicorn api:app --port 8005 --reload
```

**Frontend (React):**

```bash
cd frontend
npm i
npm run dev
```

Open your browser at the URL shown (usually `http://localhost:5173`).

## Testing

The repository includes a suite of tests in the `tests/` directory.

**Run all tests:**

```bash
python3 -m unittest discover tests
```

**Run specific tests:**

- Scenarios: `python3 tests/test_scenarios.py`
- Semantic Search: `python3 tests/test_semantic_search.py`
- Ranking Logic: `python3 tests/test_debug_ranking_v2.py`

## Features

- **Multilingual RAG:** Specialized for Hebrew text with RTL support.
- **Hybrid Search:** Combines semantic similarity with keyword boosting and bigram matching for accurate retrieval.
- **Model Agnostic:** Switch between local (Ollama) and cloud (OpenAI) models instantly.
- **Streaming:** Real-time character-by-character response streaming.
- **Citations:** Automatic citation of source documents used in the answer.
