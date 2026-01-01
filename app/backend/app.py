import streamlit as st
import requests

st.set_page_config(page_title="Enterprise AI Assistant", page_icon="🏢")
st.title("📂 Enterprise Knowledge Assistant")

# Sidebar for Ingestion
with st.sidebar:
    st.header("Admin: Upload Knowledge")
    uploaded_file = st.file_uploader("Upload PDF Policy/Manual", type="pdf")
    if st.button("Process Document"):
        if uploaded_file:
            files = {"file": uploaded_file.getvalue()}
            res = requests.post("http://localhost:8000/ingest", files={"file": (uploaded_file.name, uploaded_file.getvalue())})
            st.success(res.json()["message"])

# Chat Interface
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask me about company policies..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response = requests.get(f"http://localhost:8000/ask?query={prompt}").json()
        answer = response["answer"]
        st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})