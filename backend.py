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
    Initialize and return a MongoDB Atlas Vector Search store for RAG operations.
    
    Creates a vector store instance configured with the HuggingFace embedding model
    and MongoDB Atlas collection for similarity-based document retrieval.
    
    Returns:
        MongoDBAtlasVectorSearch: Configured vector store instance for document
            embedding storage and retrieval operations.
    
    Note:
        Uses sentence-transformers/all-mpnet-base-v2 embedding model (768 dimensions).
        Connection parameters are loaded from Streamlit secrets configuration.
    """
    client = MongoClient(MONGO_URI)
    collection = client[DB_NAME][COLLECTION_NAME]
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    
    vector_store = MongoDBAtlasVectorSearch(
        collection=collection, 
        embedding=embeddings, 
        index_name=ATLAS_VECTOR_SEARCH
    )
    return vector_store

def ingest_text(text_content):
    """
    Store text content as vector embeddings in the MongoDB Atlas collection.
    
    Args:
        text_content (str): Text content to be embedded and stored.
    
    Note:
        Documents are appended to the existing collection. Embedding generation
        is handled automatically by the configured HuggingFace model.
    """
    vector_store = get_vector_store()
    doc = Document(page_content=text_content)
    vector_store.add_documents([doc])

def get_rag_response(query):
    """
    Generate an AI response using Retrieval-Augmented Generation.
    
    Retrieves relevant documents from the vector store and generates a response
    using Google's Gemini model based on the retrieved context.
    
    Args:
        query (str): User query to be answered.
    
    Returns:
        dict: Response dictionary containing:
            - answer (str): Generated response from the LLM.
            - sources (list[Document]): Retrieved documents used for context.
    
    Note:
        Retrieves top 3 most similar documents (k=3) for context generation.
        Uses gemini-2.5-flash model for response generation.
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
    Extract vector embeddings for query and retrieved documents.
    
    Args:
        query (str): Search query to generate embeddings for.
    
    Returns:
        dict: Embedding data containing:
            - query_vector (list[float]): 768-dimensional query embedding.
            - docs (list[dict]): Retrieved documents with embeddings, each containing:
                - content (str): Document text content.
                - vector (list[float]): 768-dimensional document embedding.
                - type (str): Document type identifier.
    
    Note:
        Retrieves top 5 documents (k=5). Embedding generation may be computationally
        intensive for large document sets.
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
    Convert text to speech audio using ElevenLabs API.
    
    Args:
        text (str): Text content to convert to speech.
    
    Returns:
        bytes: MP3 audio data, or None if conversion fails.
    
    Raises:
        Displays error via Streamlit interface on API failure.
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