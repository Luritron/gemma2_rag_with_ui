from celery import shared_task
from .utils import extract_text_from_pdf, get_chroma_db_path, embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from pathlib import Path
import os

@shared_task
def process_uploaded_files(dialog_id, file_paths):
    upload_dir = Path(f"./llm_rag_project/txt_files/{dialog_id}")
    documents = []

    print(f"Started processing files for dialog {dialog_id}")
    # Обработка файлов
    for file_path in file_paths:
        file_path = Path(file_path)
        if file_path.suffix.lower() == ".pdf":
            pdf_text = extract_text_from_pdf(file_path)
            documents.append(Document(page_content=pdf_text, metadata={"source": str(file_path)}))
        elif file_path.suffix.lower() == ".txt":
            with open(file_path, "r", encoding="utf-8") as txt_file:
                text = txt_file.read()
                documents.append(Document(page_content=text, metadata={"source": str(file_path)}))

    # Разбиение текста на чанки
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, add_start_index=True)
    texts = []
    for doc in documents:
        chunks = text_splitter.split_documents([doc])
        texts.extend(chunks)

    # Сохранение в векторное хранилище
    dialog_db_path = get_chroma_db_path(dialog_id).as_posix()
    vectorstore = Chroma.from_documents(documents=texts, embedding=embeddings, persist_directory=dialog_db_path)
    print(f"Completed processing files for dialog {dialog_id}")
    return f"Indexed {len(texts)} chunks for dialog {dialog_id}."
