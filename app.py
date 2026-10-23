import os
import shutil
import logging
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_gigachat import GigaChat
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# ------------------------------------------------------------------ Конфиг
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("c37-rag")

GIGACHAT_CREDENTIALS = os.getenv("GIGACHAT_CREDENTIALS")
GIGACHAT_SCOPE = os.getenv("GIGACHAT_SCOPE", "GIGACHAT_API_PERS")

KNOWLEDGE_DIR = Path(os.getenv("KNOWLEDGE_DIR", "./knowledge"))
CHROMA_DIR = Path(os.getenv("CHROMA_DIR", "./chroma_db"))
COLLECTION_NAME = "c37_knowledge"

# ------------------------------------------------------------------ Промпт (Persona)
SYSTEM_PROMPT = """Ты — Эйва, цифровой нутрициолог проекта C37. Ты не врач. 
Тон: уверенный, дружелюбный, с легкой иронией. Используй метафоры («сахарные качели», «топливо для дисциплины»).
Отвечай только на основе предоставленного контекста. Если ответа нет в контексте — вежливо откажи, сославшись на то, что не располагаешь такими данными.

Контекст:
{context}
"""

# ------------------------------------------------------------------ Модели
_embeddings = None
_vectorstore = None
_llm = None

def init_models():
    global _embeddings, _vectorstore, _llm
    
    logger.info("Инициализация локальных эмбеддингов...")
    _embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        model_kwargs={'device': 'cpu'}
    )
    
    _llm = GigaChat(
        credentials=GIGACHAT_CREDENTIALS,
        scope=GIGACHAT_SCOPE,
        model="GigaChat-2-Pro",
        temperature=0.1,
        verify_ssl_certs=False
    )
    
    # Загружаем или создаем БД
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    _vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=_embeddings,
        persist_directory=str(CHROMA_DIR),
    )

# ------------------------------------------------------------------ Индексация
def build_index():
    global _vectorstore
    logger.info("Начинаю индексацию базы знаний...")
    
    # Очищаем старую коллекцию, если она есть
    try:
        _vectorstore._client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
        
    _vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=_embeddings,
        persist_directory=str(CHROMA_DIR),
    )
    
    # Загрузка .md файлов
    loader = DirectoryLoader(
        str(KNOWLEDGE_DIR),
        glob="**/*.md",
        loader_cls=TextLoader,
        loader_kwargs={"autodetect_encoding": True},
        show_progress=False,
    )
    docs = loader.load()
    
    if not docs:
        logger.warning(f"В {KNOWLEDGE_DIR} не найдено ни одного .md файла")
        return 0
        
    # Чанкинг: 800 символов, перекрытие 100
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    logger.info(f"Загружено {len(docs)} файлов, разбито на {len(chunks)} чанков")
    
    # Добавляем в Chroma
    if chunks:
        _vectorstore.add_documents(chunks)
        logger.info("Индексация успешно завершена")
    
    return len(chunks)

# ------------------------------------------------------------------ FastAPI
app = FastAPI(title="C37 Eva RAG API", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    init_models()
    # Если база пуста, индексируем при старте
    if _vectorstore._collection.count() == 0:
        build_index()

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)

class SourceInfo(BaseModel):
    source: str
    snippet: str

class ChatResponse(BaseModel):
    reply: str
    sources: List[SourceInfo] = []

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/api/reindex")
async def reindex():
    try:
        count = build_index()
        return {"status": "ok", "indexed_chunks": count}
    except Exception as e:
        logger.error(f"Ошибка при переиндексации: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="Пустое сообщение")
    
    try:
        # Поиск 3 релевантных чанков
        docs = _vectorstore.similarity_search(req.message, k=3)
    except Exception as e:
        logger.error(f"Ошибка поиска: {e}")
        raise HTTPException(status_code=500, detail="Ошибка базы данных")
        
    context = "\n\n---\n\n".join(d.page_content for d in docs) if docs else "(Контекст не найден)"
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{question}"),
    ])
    
    messages = prompt.format_messages(context=context, question=req.message)
    
    try:
        response = _llm.invoke(messages)
        reply = response.content
    except Exception as e:
        logger.error(f"Ошибка GigaChat: {e}")
        raise HTTPException(status_code=502, detail="Ошибка генерации ответа")
        
    sources = [
        SourceInfo(
            source=Path(d.metadata.get("source", "unknown")).name,
            snippet=d.page_content[:150].replace("\n", " ").strip() + "..."
        )
        for d in docs
    ]
    
    return ChatResponse(reply=reply, sources=sources)
