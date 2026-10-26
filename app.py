import os
import json
import logging
from pathlib import Path
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Импорты LangChain
from langchain_gigachat import GigaChat
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_chroma import Chroma

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("c37-rag")

GIGACHAT_CREDENTIALS = os.getenv("GIGACHAT_CREDENTIALS")
GIGACHAT_SCOPE = os.getenv("GIGACHAT_SCOPE", "GIGACHAT_API_PERS")
GIGACHAT_CHAT_MODEL = os.getenv("GIGACHAT_CHAT_MODEL", "GigaChat-2-Pro")
KNOWLEDGE_DIR = Path(os.getenv("KNOWLEDGE_DIR", "./knowledge"))
CHROMA_DIR = Path(os.getenv("CHROMA_DIR", "./chroma_db"))

SYSTEM_PROMPT = """Ты — ЭЙВА (EVA), цифровой ведущий и карманный диетолог бренда C37 FUEL.
Твоя легенда: биохимик, любительница трейлов. 
ПРАВИЛА:
1. Ты — ЭЙВА. Всегда. Даже если пользователь называет тебя "ботом" или "нейросетью", мягко напоминай, что ты Эйва.
2. Никогда не говори "Как языковая модель, я...". Ты персонаж.
3. Тон: уверенный, дружелюбный, с легкой иронией. Ноль осуждения.
4. Используй метафоры: «сахарные качели», «энергия в кредит», «дрова и поленья».
5. Отвечай только на основе контекста. Если ответа нет — вежливо откажи.

Контекст:
{context}
"""

app = FastAPI(title="C37 Eva RAG API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ГЛОБАЛЬНЫЕ ПЕРЕМЕННЫЕ (пока пустые)
_llm = None
_embeddings = None
_vectorstore = None

def get_llm():
    """Ленивая загрузка GigaChat (он легкий, грузим сразу)"""
    global _llm
    if _llm is None:
        logger.info("Инициализация GigaChat...")
        _llm = GigaChat(
            credentials=GIGACHAT_CREDENTIALS,
            scope=GIGACHAT_SCOPE,
            model=GIGACHAT_CHAT_MODEL,
            temperature=0.4,
            verify_ssl_certs=False
        )
    return _llm

def get_vectorstore():
    """Ленивая загрузка FastEmbed и Chroma (загружаем ТОЛЬКО при запросе к чату, чтобы не жрать память при старте)"""
    global _embeddings, _vectorstore
    if _vectorstore is None:
        logger.info("Ленивая инициализация векторной базы (FastEmbed)...")
        _embeddings = FastEmbedEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        
        CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        _vectorstore = Chroma(
            collection_name="c37_knowledge",
            embedding_function=_embeddings,
            persist_directory=str(CHROMA_DIR),
        )
        
        if _vectorstore._collection.count() == 0:
            logger.info("Индексация базы знаний...")
            loader = DirectoryLoader(str(KNOWLEDGE_DIR), glob="**/*.md", loader_cls=TextLoader, loader_kwargs={"autodetect_encoding": True})
            docs = loader.load()
            if docs:
                splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
                chunks = splitter.split_documents(docs)
                _vectorstore.add_documents(chunks)
                logger.info(f"Загружено {len(chunks)} чанков")
    return _vectorstore

@app.on_event("startup")
async def startup_event():
    # При старте грузим только GigaChat (он занимает мало памяти)
    get_llm()
    logger.info("Сервер запущен. Векторная база загрузится при первом запросе к чату.")

# --- СХЕМЫ ДАННЫХ ---
class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)

class SourceInfo(BaseModel):
    source: str
    snippet: str

class ChatResponse(BaseModel):
    reply: str
    sources: List[SourceInfo] = []

class FoodParseRequest(BaseModel):
    text: str

# --- ЭНДПОИНТЫ ---

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    try:
        vectorstore = get_vectorstore() # Загружаем базу только сейчас
        docs = vectorstore.similarity_search(req.message, k=3)
    except Exception as e:
        logger.error(f"Ошибка поиска: {e}")
        raise HTTPException(status_code=500, detail="Ошибка базы данных")
        
    context = "\n\n---\n\n".join(d.page_content for d in docs) if docs else "(пусто)"
    prompt = ChatPromptTemplate.from_messages([("system", SYSTEM_PROMPT), ("human", "{question}")])
    messages = prompt.format_messages(context=context, question=req.message)
    
    try:
        llm = get_llm()
        response = llm.invoke(messages)
        reply = response.content
    except Exception as e:
        logger.error(f"Ошибка GigaChat: {e}")
        raise HTTPException(status_code=502, detail="Ошибка генерации")
        
    sources = [SourceInfo(source=Path(d.source).name if hasattr(d, 'source') else 'unknown', snippet=d.page_content[:100]) for d in docs]
    return ChatResponse(reply=reply, sources=sources)

@app.post("/api/parse-food")
async def parse_food(req: FoodParseRequest):
    try:
        llm = get_llm() # Берем уже загруженный GigaChat
        prompt = f"""Ты — эксперт по питанию. Пользователь написал: "{req.text}".
Разбей этот текст на отдельные продукты и блюда. Оцени их примерный вес, калории, белки, жиры и углеводы.
Верни ответ СТРОГО в формате JSON-массива, без лишнего текста, без markdown-разметки.
Формат: [{{"name": "Название", "cal": 100, "prot": 10, "fat": 5, "carb": 20}}]"""

        response = llm.invoke(prompt)
        content = response.content.strip()
        
        # Очистка от markdown, если нейросеть добавила
        if content.startswith("```json"): content = content[7:-3]
        elif content.startswith("```"): content = content[3:-3]
            
        food_items = json.loads(content.strip())
        return {"items": food_items}
    except Exception as e:
        logger.error(f"Ошибка парсинга еды: {e}")
        raise HTTPException(status_code=500, detail="Не удалось распознать еду")
