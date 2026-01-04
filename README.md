# 🧠 Personal Knowledge Assistant- The Second Brain

Your AI-powered second brain that helps you store, search, and retrieve information from PDFs, notes, articles, and research papers using advanced Retrieval-Augmented Generation (RAG).

## ✨ Features

- 📄 **PDF Upload**: Extract and store content from PDF documents automatically
- ✍️ **Text Notes**: Add notes, articles, and information manually
- 🔍 **Semantic Search**: Ask questions in natural language and get AI-powered answers
- 📚 **Source Citations**: See exactly which documents were used to generate each answer
- 🏷️ **Topic Organization**: Tag content by subject for better organization
- 🔊 **Text-to-Speech**: Listen to responses with built-in audio playback
- 💬 **Chat History**: Maintain conversation context for follow-up questions
- 🆓 **100% Free**: Uses free-tier services and open-source models

## 🎯 Use Cases

Perfect for:

- 📖 Students organizing study materials and lecture notes
- 🔬 Researchers managing papers and research documentation
- 💼 Professionals building a personal knowledge base
- 📝 Anyone who wants to remember everything they've ever read

## Tech Stack

- **Frontend**: [Streamlit](https://streamlit.io/) - Interactive web interface
- **Vector Database**: MongoDB Atlas Vector Search for semantic document retrieval
- **Embeddings**: HuggingFace `sentence-transformers/all-mpnet-base-v2` (768 dimensions, runs locally)
- **LLM**: Google Gemini `gemini-2.5-flash` for natural language generation
- **PDF Processing**: [PyPDF2](https://pypdf2.readthedocs.io/en/3.x/) for text extraction from PDF files
- **Text-to-Speech**: [ElevenLabs](https://elevenlabs.io/docs/api-reference/introduction) API for audio generation
- **Framework**: [LangChain](https://www.langchain.com/) for RAG pipeline orchestration

### Why These Technologies?

✅ **Free & Cost-Effective**: All components have free tiers  
✅ **Local Embeddings**: Sentence transformers run on your machine - no API costs  
✅ **Scalable**: MongoDB Atlas handles millions of vectors efficiently  
✅ **Fast**: Gemini Flash provides quick response times  
✅ **Production-Ready**: Enterprise-grade stack suitable for real applications

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
cd Hacks-for-Hackers

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
code .streamlit/secrets.toml  # or use any text editor (vim, emacs, nano, etc.)
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

# ElevenLabs API Key (for text-to-speech)
ELEVENLABS_API_KEY = "your-elevenlabs-api-key-here"
```

**Replace the placeholders:**

- `<username>` and `<password>`: Your MongoDB Atlas credentials
- `<cluster>`: Your cluster address (e.g., `cluster0.abc123.mongodb.net`)
- `your-google-api-key-here`: Your Google AI API key ([Get one here](https://aistudio.google.com/app/apikey))
- `your-elevenlabs-api-key-here`: Your ElevenLabs API key ([Sign up here](https://elevenlabs.io/))

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

### Adding Content to Your Knowledge Base

**Option 1: Upload PDF**

1. Click on the sidebar → **📄 Upload PDF** tab
2. Select a PDF file (notes, research papers, articles, etc.)
3. Optionally add a subject/topic tag
4. Click **📤 Upload PDF**
5. Wait for confirmation

**Option 2: Enter Text Manually**

1. Click on the sidebar → **✍️ Enter Text** tab
2. Paste or type your notes/information
3. Optionally add a subject/topic tag
4. Click **📤 Add to Knowledge Base**

### Asking Questions

1. Type your question in the chat input at the bottom
2. The AI will search your knowledge base and generate an answer
3. Click **📚 Sources** to see which documents were used
4. Click **🔊 Listen to response** to hear the answer
5. Ask follow-up questions - the chat maintains context!

**Example Questions:**

- "Summarize what I learned about quantum physics"
- "What are the key points from my machine learning notes?"
- "Find information about photosynthesis"
- "What did I save about Python decorators?"

## Project Structure

```sh
├── home.py                 # Main Streamlit UI and chat interface
├── backend.py              # Core RAG logic (PDF processing, ingestion, retrieval, generation)
├── requirements.txt        # Python dependencies
├── HACKATHON_GUIDE.md     # Comprehensive guide for demo and presentation
└── .streamlit/
    └── secrets.toml        # API keys and configuration (not in repo, gitignored)
```

## Key Functions in `backend.py`

### Document Management

- **`extract_text_from_pdf(pdf_file)`**: Extracts text content from uploaded PDF files using PyPDF2
- **`ingest_text(text_content, metadata=None)`**: Converts text to embeddings and stores in MongoDB with optional metadata (subject, source type, upload date)

### RAG Pipeline

- **`get_vector_store()`**: Initializes MongoDB Atlas Vector Search with HuggingFace embeddings
- **`get_rag_response(query)`**: 
  - Retrieves top 3 most relevant documents using semantic search
  - Generates AI answer using Google Gemini with retrieved context
  - Returns both the answer and source documents
  
### Utilities

- **`text_to_speech(text)`**: Converts text responses to audio using ElevenLabs API
- **`get_vectors_for_visualization(query)`**: Extracts embeddings for analysis and debugging

### How RAG Works in This App

1. **Ingestion**: Text is split into chunks → converted to 768-dim vectors → stored in MongoDB
2. **Retrieval**: User query → converted to vector → find top K similar documents
3. **Generation**: Retrieved docs + user query → sent to Gemini → natural language answer
4. **Citation**: Source documents are returned alongside the answer

## Troubleshooting

**Connection errors?**

- Check your `MONGO_URI` in `secrets.toml`
- Ensure your IP address is whitelisted in MongoDB Atlas (or use `0.0.0.0/0` for development)

**API errors?**

- Verify your `GOOGLE_API_KEY` is valid and has not exceeded quota
- Check that `ELEVENLABS_API_KEY` is correct if using text-to-speech

**No search results?**

- Make sure your Vector Search Index is created and named correctly (`vector_index_ghw`)
- Verify you've added some content to your knowledge base first
- Check that the embedding dimension is set to 768 in the index

**Import errors?**

- Ensure all dependencies are installed: `pip install -r requirements.txt`
- Make sure you're using the correct Python environment (activate venv)

**PDF upload fails?**

- Ensure the PDF is not password-protected or corrupted
- Try a different PDF to isolate the issue
- Check that PyPDF2 is installed: `pip install PyPDF2`

**Slow responses?**

- First query may be slow as models download/initialize
- Subsequent queries should be faster
- Consider reducing `k` value in retriever (currently 3) for faster searches

## 🚀 Future Enhancements

- [ ] Chunk large documents for better retrieval accuracy
- [ ] Add filters to search by subject/date
- [ ] Support for more file types (DOCX, TXT, etc.)
- [ ] Voice input for queries
- [ ] Export chat history
- [ ] Quiz generation from stored knowledge
- [ ] Multi-user support with authentication
- [ ] Mobile-responsive design

## 📝 License

MIT License - See [LICENSE](LICENSE) file for details
