import streamlit as st
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import get_llm, DB_DIR
from langchain_google_genai import GoogleGenerativeAIEmbeddings # Import mới
from langchain_chroma import Chroma
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

st.set_page_config(page_title="Hỏi đáp", page_icon="💬", layout="wide")

with st.sidebar:
    model_choice = st.selectbox("Model", ["Gemini 2.5 Flash", "DeepSeek R1"])
    api_key = st.text_input("Google API Key", type="password") # Key này dùng cho cả LLM và Embedding

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Chào bạn!"}]

for m in st.session_state.messages:
    st.chat_message(m["role"]).markdown(m["content"])

if prompt := st.chat_input("Hỏi..."):
    if not api_key:
        st.error("Nhập API Key!")
        st.stop()
        
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").markdown(prompt)

    with st.chat_message("assistant"):
        try:
            # Dùng Google Embedding (Nhẹ RAM)
            emb_func = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
            
            if not os.path.exists(DB_DIR):
                st.error("Chưa có dữ liệu.")
                st.stop()
                
            vector_db = Chroma(persist_directory=DB_DIR, embedding_function=emb_func)
            retriever = vector_db.as_retriever(search_kwargs={"k": 10})
            
            llm = get_llm(model_choice, api_key)
            
            # ... (Phần Prompt và Chain giữ nguyên như cũ) ...
            sys_prompt = "Trả lời dựa trên context sau:\n{context}"
            chain = create_retrieval_chain(retriever, create_stuff_documents_chain(llm, ChatPromptTemplate.from_messages([("system", sys_prompt), ("human", "{input}")])))
            
            res = chain.invoke({"input": prompt})
            st.markdown(res['answer'])
            st.session_state.messages.append({"role": "assistant", "content": res['answer']})
            
        except Exception as e:
            st.error(f"Lỗi: {e}")
