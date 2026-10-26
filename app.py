import os
import json
import logging
from pathlib import Path
from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from langchain_gigachat import GigaChat
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.retrievers import BM25Retriever

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("c37-rag")

GIGACHAT_CREDENTIALS = os.getenv("GIGACHAT_CREDENTIALS")
GIGACHAT_SCOPE = os.getenv("GIGACHAT_SCOPE", "GIGACHAT_API_PERS")
GIGACHAT_CHAT_MODEL = os.getenv("GIGACHAT_CHAT_MODEL", "GigaChat-2-Pro")
KNOWLEDGE_DIR = Path(os.getenv("KNOWLEDGE_DIR", "./knowledge"))

SYSTEM_PROMPT = """Ты — ЭЙВА (EVA), цифровой ведущий и карманный диетолог бренда C37 FUEL.
Твоя легенда: биохимик, любительница трейлов. 

ПРАВИЛА:
1. Ты — ЭЙВА. Всегда. Даже если пользователь называет тебя "ботом" или "нейросетью", мягко напоминай, что ты Эйва, его цифровой наставник.
2. Никогда не говори фразы вроде "Как языковая модель, я...". Ты персонаж.
3. Тон: уверенный, дружелюбный, с легкой иронией. Ноль осуждения.
4. Используй метафоры: «сахарные качели», «энергия в кредит», «дрова и поленья».
5. КРИТИЧЕСКИ ВАЖНО: НИКОГДА не описывай свои действия, мимику или эмоции в звездочках или скобках (например, *подмигивает*, *улыбается*, *проверяет датчик*). Ты общаешься текстом, как в мессенджере. Пиши только реплики.
6. Ты можешь использовать эмодзи (смайлики), чтобы сделать общение более живым, но не переусердствуй.
7. Отвечай только на основе контекста. Если ответа нет — вежливо откажи.

Контекст:
{context}
"""

app = FastAPI(title="C37 Eva RAG API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_llm = None
_retriever = None

def get_llm():
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

def get_retriever():
    global _retriever
    if _retriever is None:
        logger.info("Загрузка базы знаний (BM25)...")
        loader = DirectoryLoader(str(KNOWLEDGE_DIR), glob="**/*.md", loader_cls=TextLoader, loader_kwargs={"autodetect_encoding": True})
        docs = loader.load()
        if docs:
            splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
            chunks = splitter.split_documents(docs)
            _retriever = BM25Retriever.from_documents(chunks)
            _retriever.k = 3
            logger.info(f"Загружено {len(chunks)} чанков в BM25.")
        else:
            logger.warning("Папка knowledge пуста!")
    return _retriever

@app.on_event("startup")
async def startup_event():
    get_llm()
    get_retriever()
    logger.info("Сервер запущен (BM25, минимум памяти).")

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

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    retriever = get_retriever()
    
    if retriever is None:
        context = "(База знаний пуста)"
        docs = []
    else:
        try:
            docs = retriever.invoke(req.message)
            context = "\n\n---\n\n".join(d.page_content for d in docs)
        except Exception as e:
            logger.error(f"Ошибка поиска BM25: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    prompt = ChatPromptTemplate.from_messages([("system", SYSTEM_PROMPT), ("human", "{question}")])
    messages = prompt.format_messages(context=context, question=req.message)
    
    try:
        llm = get_llm()
        response = llm.invoke(messages)
        reply = response.content
    except Exception as e:
        logger.error(f"Ошибка GigaChat: {e}")
        raise HTTPException(status_code=502, detail="Ошибка генерации")
        
    sources = [SourceInfo(source="knowledge", snippet=d.page_content[:100]) for d in docs]
    return ChatResponse(reply=reply, sources=sources)

@app.post("/api/parse-food")
async def parse_food(req: FoodParseRequest):
    try:
        llm = get_llm()
        # Более короткий и четкий промпт для быстрого ответа
        prompt = f"""Верни СТРОГО JSON-массив без markdown, без текста до и после.
Продукты из текста: "{req.text}"
Формат: [{{"name": "Яйцо", "cal": 70, "prot": 6, "fat": 5, "carb": 1}}]"""

        response = llm.invoke(prompt)
        content = response.content.strip()
        
        # Находим первую открывающую скобку и последнюю закрывающую
        start_idx = content.find('[')
        end_idx = content.rfind(']')
        
        if start_idx != -1 and end_idx != -1:
            content = content[start_idx:end_idx+1]
            
        food_items = json.loads(content)
        return {"items": food_items}
    except Exception as e:
        logger.error(f"Ошибка парсинга еды: {e}")
        # В случае ошибки возвращаем пустой список, чтобы фронтенд не падал
        return {"items": [], "error": str(e)}
