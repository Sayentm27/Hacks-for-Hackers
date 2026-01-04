import streamlit as st
import backend as rag

st.title("🧠 Personal Knowledge Assistant")
st.subheader("Your AI-powered second brain - store, search, and retrieve information instantly")
st.divider()

# Create a sidebar for uploading knowledge base content
with st.sidebar:
    st.header("📚 Add to Knowledge Base")
    
    # Tabs for different upload methods
    upload_tab1, upload_tab2 = st.tabs(["📄 Upload PDF", "✍️ Enter Text"])
    
    with upload_tab1:
        uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"], help="Upload PDFs of notes, articles, research papers, etc.")
        
        if uploaded_file is not None:
            st.info(f"Selected: {uploaded_file.name}")
            subject_pdf = st.text_input("Subject/Topic (optional)", key="subject_pdf", placeholder="e.g., Machine Learning, History")
            
            if st.button("📤 Upload PDF", key="upload_pdf_btn"):
                with st.spinner("Extracting text from PDF..."):
                    text_content = rag.extract_text_from_pdf(uploaded_file)
                    
                if text_content:
                    with st.spinner("Adding to knowledge base..."):
                        metadata = {
                            "source_type": "pdf",
                            "filename": uploaded_file.name,
                            "subject": subject_pdf if subject_pdf else "General"
                        }
                        rag.ingest_text(text_content, metadata=metadata)
                    st.success(f"✅ {uploaded_file.name} added successfully!")
                else:
                    st.error("Failed to extract text from PDF")
    
    with upload_tab2:
        user_text = st.text_area("Enter your notes, knowledge, or information", height=150, placeholder="Paste your notes, articles, or any information you want to store...")
        subject_text = st.text_input("Subject/Topic (optional)", key="subject_text", placeholder="e.g., Python Programming, Biology")
        
        if st.button("📤 Add to Knowledge Base", key="upload_text_btn"):
            if user_text:
                with st.spinner("Adding to knowledge base..."):
                    metadata = {
                        "source_type": "text",
                        "subject": subject_text if subject_text else "General"
                    }
                    rag.ingest_text(user_text, metadata=metadata)
                st.success("✅ Text added successfully!")
            else:
                st.warning("Please enter some text to upload.")
    
    st.divider()
    st.caption("💡 Tip: Add diverse sources for richer insights!")

st.header("💬 Ask Questions About Your Knowledge Base")

# Add example questions to help users
with st.expander("💡 Example Questions"):
    st.markdown("""
    Try asking:
    - "Summarize what I learned about quantum physics"
    - "What are the key points from my machine learning notes?"
    - "Find information about photosynthesis"
    - "What did I save about Python decorators?"
    - "Show me notes related to project management"
    """)

# Initialize the whole message history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display the chat history
for idx, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Show sources for assistant messages
        if message["role"] == "assistant" and "sources" in message:
            with st.expander("📚 Sources"):
                for i, doc in enumerate(message["sources"]):
                    # Show preview and full content in nested expander
                    preview = doc.page_content[:150] + "..." if len(doc.page_content) > 150 else doc.page_content
                    with st.expander(f"📄 Source {i+1}: {preview}"):
                        st.markdown(doc.page_content)
                        # Show metadata if available
                        if hasattr(doc, 'metadata') and doc.metadata:
                            st.caption(f"Metadata: {doc.metadata}")
        
        # Add TTS button for assistant messages
        if message["role"] == "assistant":
            if st.button("🔊 Listen to response", key=f"tts_{idx}"):
                with st.spinner("Generating audio..."):
                    audio_bytes = rag.text_to_speech(message["content"])
                    if audio_bytes:
                        st.audio(audio_bytes, format="audio/mpeg")

# Handle user input
prompt = st.chat_input("Ask me anything...")
if prompt:
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Generate rag response in here
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response_data = rag.get_rag_response(prompt)
            answer = response_data["answer"]
            sources = response_data["sources"]

        st.markdown(answer)
        
        # show sources in an expander
        with st.expander("📚 Sources"):
            for i, doc in enumerate(sources):
                # Show preview and full content in nested expander
                preview = doc.page_content[:150] + "..." if len(doc.page_content) > 150 else doc.page_content
                with st.expander(f"📄 Source {i+1}: {preview}"):
                    st.markdown(doc.page_content)
                    # Show metadata if available
                    if hasattr(doc, 'metadata') and doc.metadata:
                        st.caption(f"Metadata: {doc.metadata}")
        
        # Append the response to the message history with sources and rerun to show the button
        st.session_state.messages.append({
            "role": "assistant", 
            "content": answer,
            "sources": sources
        })
        st.rerun()  # Rerun to show the TTS button for the new message