# Hacks-for-Hackers

...

## Tech Stack

- **Vector Store**: MongoDB Atlas Vector Search for semantic document retrieval
- **Embeddings**: HuggingFace `sentence-transformers/all-mpnet-base-v2` (768 dimensions)
- **LLM**: Google Gemini `gemini-2.5-flash` for response generation
- **Framework**: [LangChain](https://www.langchain.com/) for RAG orchestration + [Streamlit](https://streamlit.io/) for the UI

## Prerequisites

Before starting, make sure you have:

1. **Python 3.8+** installed on your machine
2. A **MongoDB Atlas** account (free tier works!) with a cluster created
3. A **Google AI API key** ([Get one here](https://aistudio.google.com/app/apikey))

## Quick Setup

### 1️⃣ Clone and Install

```bash
# Clone the repository
git clone <your-repo-url>
cd rag-template

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2️⃣ Configure Secrets

All configuration is managed through Streamlit secrets. Create a `.streamlit` folder and add your credentials:

```bash
mkdir .streamlit
nano .streamlit/secrets.toml  # or use any text editor
```

Add the following to `.streamlit/secrets.toml`:

```toml
# MongoDB Atlas Configuration
MONGO_URI = "mongodb+srv://<username>:<password>@<cluster>.mongodb.net/?retryWrites=true&w=majority"
DB_NAME = "vector_store_database"
COLLECTION_NAME = "embeddings_stream"
ATLAS_VECTOR_SEARCH = "vector_index_ghw"

# Google AI API Key
GOOGLE_API_KEY = "your-google-api-key-here"
```

**Replace the placeholders:**

- `<username>` and `<password>`: Your MongoDB Atlas credentials
- `<cluster>`: Your cluster address (e.g., `cluster0.abc123.mongodb.net`)
- `your-google-api-key-here`: Your Google AI API key

### 3️⃣ Set Up MongoDB Atlas Vector Search Index

In your MongoDB Atlas cluster, create a Vector Search Index:

1. Go to your cluster → **Search** tab → **Create Search Index**

2. Choose **JSON Editor** and use this configuration:

```json
{
  "fields": [
    {
      "numDimensions": 768,
      "path": "embedding",
      "similarity": "cosine",
      "type": "vector"
    }
  ]
}
```

3. Set these values:
   - **Database**: `vector_store_database`
   - **Collection**: `embeddings_stream`  
   - **Index Name**: `vector_index_ghw`

> 💡 **Note**: The 768 dimensions match the HuggingFace embedding model used in this project.

## Running the Application

Once everything is configured, start the Streamlit app:

```bash
streamlit run home.py
```

The app will open in your browser at `http://localhost:8501`

## How to Use

...

## Project Structure

```sh
├── home.py                 # Main chat interface
├── backend.py              # Core RAG logic (ingestion, retrieval, generation)
├── requirements.txt        # Python dependencies
└── .streamlit/
    └── secrets.toml        # Configuration (not in repo)
```

## Key Functions in `backend.py`

- **`ingest_text(text_content)`**: Converts text to embeddings and stores in MongoDB
- **`get_rag_response(query)`**: Retrieves relevant docs and generates AI answer
- **`get_vectors_for_visualization(query)`**: Gets embeddings for plotting
- ...

## Troubleshooting

**Connection errors?** Check your `MONGO_URI` in `secrets.toml`

**API errors?** Verify your `GOOGLE_API_KEY` is valid

**No search results?** Make sure your Vector Search Index is created and named correctly

**Import errors?** Ensure all dependencies are installed: `pip install -r requirements.txt`
