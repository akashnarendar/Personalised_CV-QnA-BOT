# streamlit_app/main.py

import streamlit as st
import requests

st.set_page_config(page_title="Doc Q&A Chatbot")

st.title("📄 Naren's CV Knowledgebase")
st.caption("Powered by FAISS, HuggingFace & FastAPI")

question = st.text_input("Ask any question regarding Naren's experience:")
submit = st.button("Get Answer")

if submit and question:
    try:
        response = requests.post("http://localhost:8000/ask", params={"question": question})
        answer = response.json().get("answer", "No answer found.")
        st.success(f"🧠 {answer}")
    except Exception as e:
        st.error(f"Error: {e}")
