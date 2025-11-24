import streamlit as st
import sys
import os

# Fix đường dẫn import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import get_llm, DB_DIR

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

st.set_page_config(page_title="Hỏi đáp", page_icon="💬", layout="wide")
st.title("💬 Hỏi đáp Quy định Nội bộ")

with st.sidebar:
    st.header("⚙️ Cấu hình")
    model_choice = st.selectbox("Model", ["Gemini 2.5 Flash", "DeepSeek R1 (OpenRouter)"])
    api_key = st.text_input("API Key", type="password")
    
    # Tùy chỉnh độ sâu tìm kiếm
    search_k = st.slider("Độ sâu tìm kiếm (Số đoạn văn)", min_value=3, max_value=20, value=10)

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Chào bạn, tôi đã sẵn sàng tra cứu thông tin cho bạn."}]

for m in st.session_state.messages:
    st.chat_message(m["role"]).markdown(m["content"])

if prompt := st.chat_input("Nhập câu hỏi..."):
    if not api_key:
        st.error("⚠️ Chưa nhập API Key!")
        st.stop()
        
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("🔍 Đang quét cơ sở dữ liệu & Tổng hợp..."):
            try:
                if not os.path.exists(DB_DIR):
                    st.error("Chưa có dữ liệu. Vui lòng nạp file ở trang Quản lý.")
                    st.stop()
                    
                emb_func = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
                vector_db = Chroma(persist_directory=DB_DIR, embedding_function=emb_func)
                
                # Tăng số lượng đoạn văn tìm kiếm
                retriever = vector_db.as_retriever(search_kwargs={"k": search_k})
                
                llm = get_llm(model_choice, api_key)
                
                sys_prompt = (
                    "Bạn là chuyên gia tư vấn quy định nội bộ doanh nghiệp. "
                    "Trả lời câu hỏi dựa trên Context dưới đây.\n"
                    "YÊU CẦU: Trả lời chi tiết, dùng gạch đầu dòng, trích dẫn Metadata nguồn.\n"
                    "Dữ liệu tra cứu:\n{context}"
                )
                
                prompt_template = ChatPromptTemplate.from_messages([
                    ("system", sys_prompt),
                    ("human", "{input}")
                ])
                
                chain = create_retrieval_chain(
                    retriever, 
                    create_stuff_documents_chain(llm, prompt_template)
                )
                
                res = chain.invoke({"input": prompt})
                ans = res['answer']
                
                # Trích dẫn nguồn
                sources = {}
                for doc in res['context']:
                    name = doc.metadata.get('doc_name', 'Không tên')
                    dept = doc.metadata.get('department', 'N/A')
                    sources[f"{name} ({dept})"] = True
                
                if sources:
                    ans += "\n\n---\n**📚 Tài liệu tham khảo:**\n" + "\n".join([f"- {s}" for s in sources.keys()])
                
                st.markdown(ans)
                st.session_state.messages.append({"role": "assistant", "content": ans})
                
            except Exception as e:
                st.error(f"Lỗi xử lý: {e}")
