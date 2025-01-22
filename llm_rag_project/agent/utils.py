from pathlib import Path
import fitz
from langchain_community.embeddings import OllamaEmbeddings

# Функция для получения пути к базе данных Chroma
def get_chroma_db_path(dialog_id):
    base_dir = Path("./db-hormozi")
    dialog_dir = base_dir / dialog_id
    dialog_dir.mkdir(parents=True, exist_ok=True)
    return dialog_dir

# Функция для извлечения текста из PDF
def extract_text_from_pdf(file_path):
    text = ""
    with fitz.open(file_path) as pdf:
        for page in pdf:
            text += page.get_text()
    return text

# Создание объекта embeddings
embeddings = OllamaEmbeddings(model="nomic-embed-text", show_progress=False)
