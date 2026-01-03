import streamlit as st
import backend as rag

st.title("Recommendation System with RAG Pipeline")
st.subheader("Add a subheader with more explanation if you want")
st.divider()

# Create a sidebar
# Upload in here the context that goes into MongoDB (Knowledge base)
with st.sidebar:
    st.header("Upload context")
    user_text = st.text_area("Enter knowledge here", height=150)

    if st.button("upload to mongo DB"):
        if user_text:
            with st.spinner("Uploading to MongoDB..."):
                rag.ingest_text(user_text)
            st.success("Text uploaded successfully!")
        else:
            st.warning("Please enter some text to upload.")

st.header("Ask anything to the chat from our knowledge base")

# Initialize the whole message history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display the chat history
for idx, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
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
        
        # show sourses in an expander
        with st.expander("Sources"):
            for i, doc in enumerate(sources):
                st.markdown(f"**Source {i+1}:** {doc.page_content}")
        
        # Append the response to the message history and rerun to show the button
        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.rerun()  # Rerun to show the TTS button for the new message