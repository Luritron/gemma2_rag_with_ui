import requests
import re
import os
import uuid
from pathlib import Path
from tqdm import tqdm
from .models import DialogHistory
from .tasks import process_uploaded_files
from .utils import extract_text_from_pdf, get_chroma_db_path, embeddings
from django.core.files.storage import default_storage
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOllama
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain.schema import HumanMessage, Document
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

from pydantic import BaseModel
from ninja import NinjaAPI

api = NinjaAPI()

class QuestionSchema(BaseModel):
    question: str
    dialog_id: str

local_llm = 'gemma2:2b'

llm = ChatOllama(model=local_llm,
                 keep_alive="3h",
                 max_tokens=1024,
                 temperature=0)

template = """<bos><start_of_turn>user\nAnswer the question based only on the following context and extract out a meaningful answer. \
Please write in full sentences with correct spelling and punctuation. if it makes sense use lists. \
Please respond with the exact phrase "unable to find an answer" if the context does not provide an answer. Do not include any other text.\

CONTEXT: {context}

QUESTION: {question}

<end_of_turn>
<start_of_turn>model\n
ANSWER:"""
prompt = ChatPromptTemplate.from_template(template)

def build_prompt(context, question):
    return (
        f"""<bos><start_of_turn>user\nAnswer the question based only on the following context and extract out a meaningful answer. \
        Please write in full sentences with correct spelling and punctuation. if it makes sense use lists. \
        Please respond with the exact phrase "unable to find an answer" if the context does not provide an answer. Do not include any other text and gaps, spaces, symblos of \"\\n"\".\
        Just a short and fully clear "unable to find an answer" answer.\n\n\
        CONTEXT: {context}

        QUESTION: {question}

        <end_of_turn>
        <start_of_turn>model\n
        ANSWER:"""
    )

def search_online_google(question):
    api_key = "xxx"
    search_engine_id = "xxx"
    endpoint = "https://www.googleapis.com/customsearch/v1"
    params = {
        "key": "xxx",
        "cx": "xxx",
        "q": question,
        "num": 5,
    }

    response = requests.get(endpoint, params=params)
    if response.status_code == 200:
        results = response.json()
        snippets = [item["snippet"] for item in results.get("items", [])]
        return "\n".join(snippets)
    else:
        return "No relevant information found online."

def is_rag_answer_unavailable(answer):
    negative_responses = [
        "unable to find an answer",
        "does not contain information",
        "unable to provide information",
    ]
    return any(phrase in answer.lower() for phrase in negative_responses)

def format_llm_answer(answer):
    answer = re.sub(r"^(##)(\s+)(.*)$", r"<h2>\3</h2>", answer, flags=re.MULTILINE)
    answer = re.sub(r"^(#)(\s+)(.*)$", r"<h1>\3</h1>", answer, flags=re.MULTILINE)

    answer = re.sub(r"\*\*(.*?)\*\*", r"<strong>\1</strong>", answer)

    answer = re.sub(r"(\S)\s*\* ", r"\1<br>* ", answer)
    answer = re.sub(r"^\*\s*(.*)$", r"<br>* \1", answer, flags=re.MULTILINE)

    answer = re.sub(r"^\d+\.\s*(.*)$", r"<br>1. \1", answer, flags=re.MULTILINE)

    answer = re.sub(r"\n\n", r"</p><p>", answer)
    answer = f"<p>{answer}</p>"

    return answer

def build_prompt_with_history(messages, context, question):
    history_content = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])
    prompt = (
        f"Conversation History:\n{history_content}\n\n"
        f"Context:\n{context}\n\n"
        f"Question:\n{question}\n\n"
        f"Please provide a detailed and helpful answer."
    )
    return prompt

@api.get("/dialogs")
def list_dialogs(request):
    user_id = request.headers.get("X-User-ID", "default_user")
    dialogs = (
        DialogHistory.objects.filter(user_id=user_id)
        .values("dialog_id")
        .distinct()
        .order_by("-timestamp")
    )
    return {"dialogs": [dialog["dialog_id"] for dialog in dialogs]}

@api.post("/dialogs/new")
def start_new_dialog(request):
    user_id = request.headers.get("X-User-ID", "default_user")
    dialog_id = str(uuid.uuid4())
    DialogHistory.objects.create(user_id=user_id, dialog_id=dialog_id, role="system", content="New dialog started.")

    get_chroma_db_path(dialog_id)
    return {"dialog_id": dialog_id}

@api.get("/dialogs/{dialog_id}")
def get_dialog_messages(request, dialog_id: str):
    user_id = request.headers.get("X-User-ID", "default_user")
    messages = DialogHistory.objects.filter(user_id=user_id, dialog_id=dialog_id).order_by("timestamp")
    return {
        "messages": [
            {"role": message.role, "content": format_llm_answer(message.content), "timestamp": message.timestamp}
            for message in messages
        ]
    }


@api.post("/upload_files")
def upload_files(request):
    dialog_id = request.POST.get("dialog_id")
    uploaded_files = request.FILES.getlist("files")
    upload_dir = Path(f"./llm_rag_project/txt_files/{dialog_id}")
    upload_dir.mkdir(parents=True, exist_ok=True)

    file_paths = []
    for uploaded_file in uploaded_files:
        file_path = upload_dir / uploaded_file.name
        with open(file_path, 'wb') as f:
            for chunk in uploaded_file.chunks():
                f.write(chunk)
        file_paths.append(str(file_path))

    process_uploaded_files.delay(dialog_id, file_paths)
    return {"status": "Files uploaded and indexing started."}

@api.post("/ask")
def ask_question(request, payload: QuestionSchema):
    question = payload.question
    dialog_id = payload.dialog_id
    dialog_db_path = get_chroma_db_path(dialog_id).as_posix()
    if not os.path.exists(dialog_db_path):
        print(f"Database path does not exist: {dialog_db_path}")
    db = Chroma(persist_directory=dialog_db_path, embedding_function=embeddings)
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 5})
    user_id = request.headers.get("X-User-ID", "default_user")

    if question.startswith("@plan"):
        topic = question.replace("@plan", "").strip()
        question = f"Please create a detailed learning plan for the topic: {topic}."
    elif question.startswith("@test"):
        topic = question.replace("@test", "").strip()
        question = f"Please create a test with questions and answers for the topic: {topic}."

    results = retriever.get_relevant_documents(question)
    print(f"\nRetrieved {len(results)} relevant documents:")

    if not dialog_id:
        return {"error": "Dialog ID is required."}

    DialogHistory.objects.create(user_id=user_id, dialog_id=dialog_id, role="user", content=question)

    dialog_history = DialogHistory.objects.filter(
        user_id=user_id, dialog_id=dialog_id
    ).order_by("timestamp")
    messages = [{"role": entry.role, "content": entry.content} for entry in dialog_history]

    rag_chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
    )

    rag_answer = ""
    for chunk in rag_chain.stream(question):
        rag_answer += chunk.content

    if is_rag_answer_unavailable(rag_answer):
        print("No relevant data found in the database. Searching online...")
        internet_context = search_online_google(question)
        if internet_context:
            internet_prompt = build_prompt_with_history(messages, internet_context, question)
            try:
                response = llm.generate([[HumanMessage(content=internet_prompt)]])
                internet_answer = response.generations[0][0].text.strip()

                if is_rag_answer_unavailable(internet_answer):
                    print(f"Internet Answer is invalid: {internet_answer}")
                    return {"source": "none", "answer": "No relevant information found online."}

                DialogHistory.objects.create(user_id=user_id, dialog_id=dialog_id, role="model", content=internet_answer)

                print(f"Internet Answer: {internet_answer}")
                formatted_answer = format_llm_answer(internet_answer)
                return {"source": "internet", "answer": formatted_answer}
            except Exception as e:
                print(f"Error during LLM generation: {e}")
                return {"source": "none", "answer": "Failed to generate an answer from internet context."}
        else:
            return {"source": "none", "answer": "No relevant information found online."}

    try:
        db_prompt = build_prompt_with_history(messages, rag_answer, question)
        db_response = llm.generate([[HumanMessage(content=db_prompt)]])
        db_answer = db_response.generations[0][0].text.strip()

        if is_rag_answer_unavailable(db_answer):
            print(f"Database Answer is invalid: {db_answer}")
            return {"source": "none", "answer": "No relevant information found online."}

        DialogHistory.objects.create(user_id=user_id, dialog_id=dialog_id, role="model", content=db_answer)

        print(f"Database Answer: {db_answer}")
        formatted_answer = format_llm_answer(db_answer)
        return {"source": "database", "answer": formatted_answer}
    except Exception as e:
        print(f"Error during LLM generation: {e}")
        return {"source": "none", "answer": "Failed to generate an answer from database context."}
