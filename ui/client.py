import streamlit as st
import requests
import io

API_URL = "http://localhost:8000/api/v1"

st.title("Document Chat Application")

# File upload
st.sidebar.header("Upload Document")
uploaded_file = st.sidebar.file_uploader("Choose a PDF or Word file", type=["pdf", "docx"])

if uploaded_file is not None:
    st.sidebar.text("Uploading and processing document...")
    with st.spinner("Uploading and processing document..."):
        files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}
        response = requests.post(f"{API_URL}/upload", files=files)
        if response.status_code == 200:
            st.sidebar.success("Document uploaded and processed successfully!")
        else:
            st.sidebar.error("Error uploading document. Please try again.")

# Chat interface
st.header("Chat with Documents")

# Display chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask a question about the documents"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = requests.post(f"{API_URL}/chat", json={"question": prompt})
            if response.status_code == 200:
                answer = response.json()["answer"]
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
            else:
                st.error("Error generating response. Please try again.")
