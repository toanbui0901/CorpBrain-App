import streamlit as st

st.set_page_config(page_title="CorpBrain Portal", page_icon="🏢")

st.write("# 🏢 CorpBrain Portal")
st.info("Hệ thống quản trị tri thức & Hỏi đáp nội bộ")

st.markdown(
    """
    ### Hướng dẫn sử dụng:
    
    1. **📂 Backend Quản lý**: 
       - Dành cho Admin.
       - Tải lên văn bản quy phạm, Quyết định, Chính sách.
       
    2. **💬 Frontend Hỏi đáp**:
       - Dành cho Nhân viên.
       - Chat với AI để tra cứu thông tin chính xác từ kho dữ liệu.
    """
)
