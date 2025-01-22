from pathlib import Path
import fitz
from langchain_community.embeddings import OllamaEmbeddings

def get_chroma_db_path(dialog_id):
    base_dir = Path("./db-hormozi")
    dialog_dir = base_dir / dialog_id
    dialog_dir.mkdir(parents=True, exist_ok=True)
    return dialog_dir

def extract_text_from_pdf(file_path):
    text = ""
    with fitz.open(file_path) as pdf:
        for page in pdf:
            text += page.get_text()
    return text

embeddings = OllamaEmbeddings(model="nomic-embed-text", show_progress=False)
