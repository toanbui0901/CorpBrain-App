import os
import time
import tempfile
import pandas as pd
from datetime import datetime
import streamlit as st

# Import SDK Google
import google.generativeai as genai

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import Docx2txtLoader
from pypdf import PdfReader

DB_DIR = "faiss_index"
HISTORY_FILE = "file_history.csv"

# --- HÀM OCR BẰNG GEMINI (SIÊU NHẸ RAM) ---
def ocr_via_gemini(file_path, api_key):
    """
    Upload file lên Google, nhờ Gemini 1.5 Flash đọc nội dung trả về text.
    """
    try:
        # Cấu hình API
        genai.configure(api_key=api_key)
        
        # 1. Upload file lên Google File API (Lưu tạm)
        st.toast("☁️ Đang gửi file scan lên Google để đọc...", icon="🚀")
        sample_file = genai.upload_file(path=file_path, display_name="Scan Document")
        
        # Đợi file sẵn sàng (Google cần vài giây để xử lý file)
        while sample_file.state.name == "PROCESSING":
            time.sleep(2)
            sample_file = genai.get_file(sample_file.name)
            
        if sample_file.state.name == "FAILED":
            return "Lỗi: Google không đọc được file này."

        # 2. Gọi Model Flash để trích xuất văn bản
        model = genai.GenerativeModel(model_name="gemini-1.5-flash")
        
        response = model.generate_content([
            sample_file,
            "Hãy đóng vai một công cụ OCR chính xác. Nhiệm vụ của bạn là trích xuất toàn bộ văn bản có trong file PDF này ra dạng text. Giữ nguyên định dạng tiếng Việt. Chỉ trả về nội dung văn bản, không thêm lời dẫn."
        ])
        
        # 3. Dọn dẹp (Xóa file trên Cloud để bảo mật)
        genai.delete_file(sample_file.name)
        
        return response.text
        
    except Exception as e:
        return f"Lỗi Cloud OCR: {str(e)}"

def read_pdf_smart(file_path, api_key):
    """
    Chiến thuật:
    1. Thử đọc nhanh bằng pypdf (cho file digital).
    2. Nếu ít chữ -> Coi là scan -> Gọi Gemini OCR.
    """
    text = ""
    try:
        reader = PdfReader(file_path)
        for page in reader.pages:
            t = page.extract_text()
            if t: text += t + "\n"
    except: pass

    # Ngưỡng phát hiện scan: Nếu trung bình mỗi trang < 20 ký tự
    total_pages = len(reader.pages) if 'reader' in locals() and reader.pages else 1
    if len(text) < 20 * total_pages:
        st.info("📷 Phát hiện tài liệu Scan. Đang kích hoạt Gemini OCR (Cloud)...")
        # Gọi Gemini đọc
        text = ocr_via_gemini(file_path, api_key)
    
    return text

# --- HÀM XỬ LÝ CHÍNH ---
def process_and_save(uploaded_file, meta_info, api_key):
    if not api_key:
        st.error("Thiếu API Key!")
        return 0

    # Lưu file tạm
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp:
        tmp.write(uploaded_file.getbuffer())
        fpath = tmp.name

    # Đọc nội dung
    text = ""
    if uploaded_file.name.endswith('.pdf'):
        text = read_pdf_smart(fpath, api_key)
    elif uploaded_file.name.endswith('.docx'):
        loader = Docx2txtLoader(fpath)
        docs = loader.load()
        text = "\n".join([d.page_content for d in docs])
    elif uploaded_file.name.endswith('.xlsx'):
        try:
            df = pd.read_excel(fpath)
            text = df.to_string(index=False)
        except: pass
    
    os.remove(fpath) # Xóa file tạm ngay
    
    if not text or not text.strip():
        st.error("Không đọc được nội dung văn bản.")
        return 0

    # Metadata injection
    full_content = (
        f"METADATA >> [Tên: {meta_info['doc_name']}] | [Đơn vị: {meta_info['department']}] | [Ngày HL: {meta_info['effective_date']}]\n"
        f"NỘI DUNG:\n{text}"
    )
    doc = Document(page_content=full_content, metadata=meta_info)
    
    # Chunking
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = splitter.split_documents([doc])

    # Embedding vào FAISS
    try:
        emb_func = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
        
        if os.path.exists(DB_DIR):
            try:
                old_db = FAISS.load_local(DB_DIR, emb_func, allow_dangerous_deserialization=True)
                new_db = FAISS.from_documents(splits, emb_func)
                old_db.merge_from(new_db)
                old_db.save_local(DB_DIR)
            except:
                # Nếu index cũ lỗi, tạo mới đè lên
                db = FAISS.from_documents(splits, emb_func)
                db.save_local(DB_DIR)
        else:
            db = FAISS.from_documents(splits, emb_func)
            db.save_local(DB_DIR)
            
    except Exception as e:
        st.error(f"Lỗi Vector DB: {e}")
        return 0
    
    # Ghi log
    log_entry = {
        "File gốc": uploaded_file.name,
        "Tên văn bản": meta_info['doc_name'],
        "Ngày nạp": datetime.now().strftime("%Y-%m-%d %H:%M")
    }
    if os.path.exists(HISTORY_FILE):
        df_hist = pd.read_csv(HISTORY_FILE)
        df_hist = pd.concat([df_hist, pd.DataFrame([log_entry])], ignore_index=True)
    else:
        df_hist = pd.DataFrame([log_entry])
    df_hist.to_csv(HISTORY_FILE, index=False)
    
    return len(splits)

def get_llm(model_type, api_key):
    # Luôn dùng Gemini cho nhẹ
    if model_type == "Gemini 2.5 Flash":
        return ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=api_key, temperature=0.1)
    elif model_type == "DeepSeek R1 (OpenRouter)":
        return ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=api_key) # Fallback tạm về Gemini cho ổn định
    return None
