import os
import tempfile
import pandas as pd
from datetime import datetime
import streamlit as st

# --- KHU VỰC IMPORT ---
try:
    from langchain_openai import ChatOpenAI
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_chroma import Chroma
    from langchain_core.documents import Document
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.document_loaders import Docx2txtLoader
    from pypdf import PdfReader
    from pdf2image import convert_from_path
    import pytesseract
except ImportError:
    st.error("Thiếu thư viện! Hãy chạy: pip install langchain-chroma langchain-huggingface langchain-google-genai langchain-openai")
    st.stop()

# --- CẤU HÌNH HỆ THỐNG ---
TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
POPPLER_PATH = r"C:\Program Files\poppler-24.08.0\Library\bin"

DB_DIR = "vector_db"
HISTORY_FILE = "file_history.csv"

# Cấu hình Tesseract
if os.path.exists(TESSERACT_PATH):
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

def extract_text_with_ocr(file_path):
    """Đọc file PDF, tự động chuyển sang OCR nếu là file scan"""
    text = ""
    try:
        reader = PdfReader(file_path)
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted: text += extracted + "\n"
    except: pass

    # Logic phát hiện file scan
    total_pages = len(reader.pages) if 'reader' in locals() and reader.pages else 1
    if len(text) < 50 * total_pages:
        st.toast("📷 Đang chạy OCR (Đọc ảnh)...", icon="⏳")
        try:
            if not os.path.exists(POPPLER_PATH):
                return "Lỗi: Chưa cấu hình đúng đường dẫn Poppler."
            images = convert_from_path(file_path, dpi=300, poppler_path=POPPLER_PATH)
            ocr_text = ""
            for img in images:
                ocr_text += pytesseract.image_to_string(img, lang='vie+eng') + "\n"
            return ocr_text
        except Exception as e:
            return f"Lỗi OCR: {e}"
    return text

def process_and_save(uploaded_file, meta_info):
    """Xử lý file và lưu vào Vector DB"""
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp:
        tmp.write(uploaded_file.getbuffer())
        fpath = tmp.name

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
        f"METADATA >> [Tên VB: {meta_info['doc_name']}] | [Bộ phận: {meta_info['department']}] | [Ngày HL: {meta_info['effective_date']}]\n"
        f"NỘI DUNG VĂN BẢN:\n{text}"
    )
    doc = Document(page_content=full_content, metadata=meta_info)
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=300)
    splits = splitter.split_documents([doc])

    emb_func = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    vector_db = Chroma(persist_directory=DB_DIR, embedding_function=emb_func)
    vector_db.add_documents(splits)
    
    log_entry = {
        "File gốc": uploaded_file.name,
        "Tên văn bản": meta_info['doc_name'],
        "Đơn vị": meta_info['department'],
        "Ngày hiệu lực": str(meta_info['effective_date']),
        "Thời gian nạp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "Số đoạn": len(splits)
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
        return ChatOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
            model="deepseek/deepseek-r1:free",
            temperature=0.3
        )
    elif model_type == "Gemini 2.5 Flash":
        return ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=api_key,
            temperature=0.1
        )
    return None
