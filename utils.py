import os
import sys
import platform
import streamlit as st

# --- FIX LỖI SQLITE (BẮT BUỘC Ở ĐẦU) ---
if platform.system() != "Windows":
    try:
        __import__('pysqlite3')
        sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
    except ImportError:
        pass
# ---------------------------------------

import tempfile
import pandas as pd
from datetime import datetime

try:
    from langchain_openai import ChatOpenAI
    from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
    from langchain_chroma import Chroma
    from langchain_core.documents import Document
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.document_loaders import Docx2txtLoader
    from pypdf import PdfReader
    from pdf2image import convert_from_path
    import pytesseract
except ImportError:
    st.error("Thiếu thư viện! Kiểm tra requirements.txt")
    st.stop()

# --- CẤU HÌNH ĐƯỜNG DẪN ---
if platform.system() == "Windows":
    TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    POPPLER_PATH = r"C:\Program Files\poppler-24.08.0\Library\bin"
    if os.path.exists(TESSERACT_PATH):
        pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH
else:
    TESSERACT_PATH = "tesseract"
    POPPLER_PATH = None

DB_DIR = "vector_db"
HISTORY_FILE = "file_history.csv"

# --- HÀM HỖ TRỢ OCR ---
def extract_text_with_ocr(file_path):
    text = ""
    try:
        reader = PdfReader(file_path)
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted: text += extracted + "\n"
    except: pass

    # Nếu ít chữ quá thì coi là file scan
    if len(text) < 50:
        st.toast("📷 Đang OCR trên Cloud...", icon="☁️")
        try:
            if platform.system() == "Windows":
                images = convert_from_path(file_path, dpi=200, poppler_path=POPPLER_PATH) # Giảm DPI xuống 200 cho nhẹ RAM
            else:
                images = convert_from_path(file_path, dpi=200)
            
            ocr_text = ""
            for img in images:
                ocr_text += pytesseract.image_to_string(img, lang='vie+eng') + "\n"
            return ocr_text
        except Exception as e:
            return f"Lỗi OCR: {e}"
    return text

# --- HÀM XỬ LÝ CHÍNH (SỬ DỤNG GOOGLE EMBEDDING) ---
def process_and_save(uploaded_file, meta_info, api_key):
    """
    Cần truyền thêm api_key vào để embedding
    """
    if not api_key:
        st.error("Cần nhập API Key để xử lý dữ liệu!")
        return 0

    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp:
        tmp.write(uploaded_file.getbuffer())
        fpath = tmp.name

    # Đọc file
    text = ""
    if uploaded_file.name.endswith('.pdf'):
        text = extract_text_with_ocr(fpath)
    elif uploaded_file.name.endswith('.docx'):
        loader = Docx2txtLoader(fpath)
        docs = loader.load()
        text = "\n".join([d.page_content for d in docs])
    elif uploaded_file.name.endswith('.xlsx'):
        try:
            df = pd.read_excel(fpath)
            text = df.to_string(index=False)
        except: pass
    
    os.remove(fpath)
    if not text.strip(): return 0

    full_content = (
        f"METADATA >> [Tên: {meta_info['doc_name']}] | [Đơn vị: {meta_info['department']}] | [Ngày HL: {meta_info['effective_date']}]\n"
        f"NỘI DUNG:\n{text}"
    )
    doc = Document(page_content=full_content, metadata=meta_info)
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = splitter.split_documents([doc])

    # [THAY ĐỔI QUAN TRỌNG] Dùng Google Embedding thay vì HuggingFace (Tiết kiệm 500MB RAM)
    try:
        emb_func = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
        
        # Reset DB nếu lỗi
        try:
            vector_db = Chroma(persist_directory=DB_DIR, embedding_function=emb_func)
        except:
            import shutil
            if os.path.exists(DB_DIR): shutil.rmtree(DB_DIR)
            vector_db = Chroma(persist_directory=DB_DIR, embedding_function=emb_func)

        vector_db.add_documents(splits)
    except Exception as e:
        st.error(f"Lỗi Embedding (Kiểm tra API Key): {e}")
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
    if model_type == "DeepSeek R1 (OpenRouter)":
        return ChatOpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key, model="deepseek/deepseek-r1:free", temperature=0.3)
    elif model_type == "Gemini 2.5 Flash":
        return ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=api_key, temperature=0.1)
    return None
