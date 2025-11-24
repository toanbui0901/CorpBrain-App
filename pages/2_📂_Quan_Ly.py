import streamlit as st
import pandas as pd
import os
from datetime import date
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import process_and_save, HISTORY_FILE

st.set_page_config(page_title="Quản lý", page_icon="📂", layout="wide")
st.title("📂 Quản lý & Nạp dữ liệu")

# Thêm ô nhập API Key cho Admin để dùng Embedding
api_key = st.sidebar.text_input("Nhập Google API Key (Để Embedding)", type="password")

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("📥 Upload")
    with st.form("upload"):
        uploaded_file = st.file_uploader("File", type=["pdf", "docx", "xlsx"])
        doc_name = st.text_input("Tên văn bản")
        dept = st.selectbox("Đơn vị", ["Ban Giám Đốc", "HCNS", "Kế Toán", "Khác"])
        eff_date = st.date_input("Ngày hiệu lực", date.today())
        
        if st.form_submit_button("Lưu"):
            if not api_key:
                st.error("⚠️ Cần nhập Google API Key bên trái để xử lý!")
            elif uploaded_file and doc_name:
                with st.spinner("Đang xử lý (API Cloud)..."):
                    meta = {"doc_name": doc_name, "department": dept, "effective_date": eff_date}
                    # Truyền API Key vào hàm
                    c = process_and_save(uploaded_file, meta, api_key)
                    if c: st.success(f"✅ Xong! {c} chunks.")
            else:
                st.error("Thiếu thông tin.")

with col2:
    if os.path.exists(HISTORY_FILE):
        st.dataframe(pd.read_csv(HISTORY_FILE), use_container_width=True)
