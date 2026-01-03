import streamlit as st
from pymongo import MongoClient
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from elevenlabs.client import ElevenLabs


# Configuration for MongoDB
MONGO_URI = st.secrets["MONGO_URI"]
DB_NAME = st.secrets["DB_NAME"]
COLLECTION_NAME = st.secrets["COLLECTION_NAME"]
ATLAS_VECTOR_SEARCH = st.secrets["ATLAS_VECTOR_SEARCH"]

# Configuration for models
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
GENERATIVE_AI_MODEL = "gemini-2.5-flash"

# Initialize ElevenLabs client for TTS 
ELEVENLABS_API_KEY = st.secrets["ELEVENLABS_API_KEY"]
client = ElevenLabs(api_key=ELEVENLABS_API_KEY)
# Configure voice and model for TTS
VOICE_ID = "flHkNRp1BlvT73UL6gyz"
MODEL_ID = "eleven_turbo_v2_5"

def get_vector_store():
    """
    Initialize and return a MongoDB Atlas Vector Search store for RAG (Retrieval-Augmented Generation).
    
    This function sets up the connection to your MongoDB database and creates a vector store
    that can convert text into embeddings (numerical representations) and perform similarity searches.
    Think of it as creating a "smart search engine" that understands meaning, not just keywords.
    
    What it does step-by-step:
    1. Connects to MongoDB Atlas using your connection string
    2. Accesses the specific database and collection where embeddings are stored
    3. Initializes the HuggingFace embedding model (converts text to vectors)
    4. Creates a vector search interface that ties everything together
    
    Returns:
        MongoDBAtlasVectorSearch: A vector store object that you can use to:
            - Store text as embeddings (via add_documents)
            - Search for similar content (via similarity_search or as_retriever)
    
    Configuration used:
        - MongoDB URI: From Streamlit secrets (keeps credentials secure)
        - Database: vector_store_database
        - Collection: embeddings_stream (where your document embeddings live)
        - Vector Index: vector_index_ghw (MongoDB Atlas search index name)
        - Embedding Model: sentence-transformers/all-mpnet-base-v2 (768-dimensional vectors)
    
    Example:
        vector_store = get_vector_store()
        # Now you can search: vector_store.similarity_search("your query")
    
    Note:
        This function is called by other functions like ingest_text() and get_rag_response()
        to ensure they all use the same vector store configuration.
    """
    client = MongoClient(MONGO_URI)
    collection = client[DB_NAME][COLLECTION_NAME]
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    # embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    vector_store = MongoDBAtlasVectorSearch(
        collection=collection, 
        embedding=embeddings, 
        index_name=ATLAS_VECTOR_SEARCH
    )
    return vector_store

def ingest_text(text_content):
    """
    Store text content in the vector database as embeddings.
    
    This is how you "teach" your RAG system new information. When you call this function,
    it takes your text, converts it into a vector (numerical representation), and stores
    it in MongoDB so it can be searched later.
    
    Think of it like adding a book to a library - the text gets indexed so you can
    find it again when asking questions.
    
    Args:
        text_content (str): The text you want to store. Can be anything: a paragraph,
                           an article, documentation, etc.
    
    Returns:
        None: The function stores the data but doesn't return anything.
    
    What happens internally:
        1. Gets the vector store connection
        2. Wraps your text in a Document object (LangChain format)
        3. Converts the text to embeddings and saves to MongoDB
    
    Example:
        ingest_text("Python is a programming language created by Guido van Rossum.")
        # Now your RAG system can answer questions about Python!
    
    Note:
        - Each call adds NEW documents to the database (doesn't replace existing ones)
        - The embedding process happens automatically via the HuggingFace model
        - Used by the Streamlit app when you paste text in the "Ingest Documents" tab
    """
    vector_store = get_vector_store()
    doc = Document(page_content=text_content)
    vector_store.add_documents([doc])

def get_rag_response(query):
    """
    Generate an AI answer to a query using Retrieval-Augmented Generation (RAG).
    
    This is the CORE function of your RAG system! It combines:
    1. Retrieval: Finding relevant documents from your vector database
    2. Generation: Using an LLM (Gemini) to generate an answer based on those documents
    
    Think of it like asking a smart assistant that can search your personal library
    before answering, instead of just making things up.
    
    Args:
        query (str): The question you want to ask. Examples:
                    "What is Python?"
                    "How does machine learning work?"
    
    Returns:
        dict: A dictionary containing:
            - "answer" (str): The AI-generated response to your query
            - "sources" (list): The actual documents that were used to generate the answer
                              (so you can verify the information)
    
    What happens step-by-step:
        1. Connects to the vector store
        2. Initializes Google's Gemini AI model (gemini-2.5-flash)
        3. Creates a retriever that finds the top 3 most relevant documents (k=3)
        4. Searches your database for documents similar to the query
        5. Combines all retrieved documents into a single context string
        6. Creates a prompt template with system instructions and the user question
        7. Builds a chain: prompt → LLM → text parser
        8. Sends everything to Gemini and gets back an answer
        9. Returns both the answer and source documents
    
    Example:
        result = get_rag_response("What is Python?")
        print(result["answer"])  # AI's answer
        print(result["sources"])  # Documents it used to answer
    
    Note:
        - Uses k=3 to retrieve top 3 most similar documents (adjustable in search_kwargs)
        - The LLM only sees the context from YOUR documents, not its general knowledge
        - Sources let you fact-check and see where the answer came from
        - The "|" operator chains LangChain components together (pipe operator)
    """
    vector_store = get_vector_store()

    llm = ChatGoogleGenerativeAI(model=GENERATIVE_AI_MODEL)
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    docs = retriever.invoke(query)

    context_text = "\n\n".join([doc.page_content for doc in docs])

    prompt = ChatPromptTemplate.from_messages([
        ("system", "Use the following context to answer:\n\n{context}"),
        ("human", "{question}")
    ])

    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({"context": context_text, "question": query})
    return {
        "answer": answer, 
        "sources": docs
    }

def get_vectors_for_visualization(query):
    """
    Get vector embeddings for visualization purposes.
    
    This function helps you SEE how your RAG system works "under the hood" by extracting
    the actual vector embeddings (numerical representations) of your query and the top
    matching documents. This is useful for debugging or understanding vector similarity.
    
    Think of it like x-raying your search - you get to see the math that makes it work!
    
    Args:
        query (str): The search query you want to visualize. Example: "What is Python?"
    
    Returns:
        dict: A dictionary containing:
            - "query_vector" (list): The 768-dimensional vector of your query
            - "docs" (list): List of dictionaries, each containing:
                - "content" (str): The actual text of the document
                - "vector" (list): The 768-dimensional vector of that document
                - "type" (str): Always "Document" (for categorization)
    
    What happens step-by-step:
        1. Connects to the vector store
        2. Gets the embeddings model (same one used for storage)
        3. Converts your query into a vector (768 numbers)
        4. Retrieves the top 5 most similar documents (k=5)
        5. For each document, converts its text into a vector
        6. Packages everything into a structured format for visualization
    
    Example:
        data = get_vectors_for_visualization("What is Python?")
        print(len(data["query_vector"]))  # 768 (dimensions)
        print(len(data["docs"]))  # Up to 5 documents
        print(data["docs"][0]["content"])  # Text of most similar document
    
    Note:
        - Uses k=5 to get top 5 documents (more than RAG response for better viz)
        - Each vector has 768 dimensions (specific to all-mpnet-base-v2 model)
        - Used by the Streamlit app's "Visualize Vectors" feature
        - Vectors with similar numbers are semantically similar in meaning
        - This is a SLOW operation - converting text to vectors takes time!
    """
    vector_store = get_vector_store()
    embeddings = vector_store.embeddings
    query_vector = embeddings.embed_query(query)

    retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    docs = retriever.invoke(query)

    doc_data = []
    for doc in docs:
        vec = embeddings.embed_query(doc.page_content)
        doc_data.append({
            "content": doc.page_content,
            "vector": vec,
            "type": "Document"
        })
    return {
        "query_vector": query_vector,
        "docs": doc_data
    }


def text_to_speech(text):
    """
    Convert text to speech using ElevenLabs API.
    
    Args:
        text (str): The text to convert to speech
    
    Returns:
        bytes: Audio data in MP3 format, or None if error occurs
    """
    try:
        client = ElevenLabs(api_key=ELEVENLABS_API_KEY)
        audio_generator = client.text_to_speech.convert(
            text=text,
            voice_id=VOICE_ID,
            model_id=MODEL_ID
        )
        return b"".join(audio_generator)
    except Exception as e:
        st.error(f"TTS Error: {str(e)}")
        return None