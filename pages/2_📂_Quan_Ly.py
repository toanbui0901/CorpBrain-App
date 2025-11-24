import streamlit as st
import pandas as pd
import os
from datetime import date
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import process_and_save, HISTORY_FILE

st.set_page_config(page_title="Quản lý", page_icon="📂", layout="wide")
st.title("📂 Quản lý Cơ sở Dữ liệu Văn bản")

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("📥 Nạp dữ liệu")
    with st.form("upload_form", clear_on_submit=True):
        uploaded_file = st.file_uploader("File (PDF/DOCX/XLSX)", type=["pdf", "docx", "xlsx"])
        doc_name = st.text_input("Tên văn bản (VD: QĐ 01/2024)")
        dept = st.selectbox("Đơn vị", ["Ban Giám Đốc", "HCNS", "Kế Toán", "Kinh Doanh", "IT"])
        eff_date = st.date_input("Ngày hiệu lực", date.today())
        
        if st.form_submit_button("Lưu vào Hệ thống"):
            if uploaded_file and doc_name:
                with st.spinner("Đang xử lý..."):
                    meta = {"doc_name": doc_name, "department": dept, "effective_date": eff_date}
                    c = process_and_save(uploaded_file, meta)
                    if c: st.success(f"✅ Xong! {c} chunks.")
                    else: st.error("Lỗi đọc file.")
            else:
                st.error("Thiếu thông tin.")

with col2:
    st.subheader("🗃️ Nhật ký tải lên")
    if os.path.exists(HISTORY_FILE):
        df = pd.read_csv(HISTORY_FILE)
        st.dataframe(df, use_container_width=True)
    else:
        st.info("Chưa có dữ liệu.")
