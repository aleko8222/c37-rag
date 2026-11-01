import os
import json
import logging
import re
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from langchain_gigachat import GigaChat
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.retrievers import BM25Retriever

# Настройка логирования
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

# === БАЗА ПРОДУКТОВ (на 100 грамм) ===
FOOD_DB = {
    # Белки
    "курица": {"cal": 165, "prot": 31, "fat": 3.6, "carb": 0, "default_g": 150},
    "куриная грудка": {"cal": 165, "prot": 31, "fat": 3.6, "carb": 0, "default_g": 150},
    "индейка": {"cal": 104, "prot": 21, "fat": 1.5, "carb": 0, "default_g": 150},
    "говядина": {"cal": 250, "prot": 26, "fat": 16, "carb": 0, "default_g": 150},
    "свинина": {"cal": 259, "prot": 27, "fat": 16, "carb": 0, "default_g": 150},
    "рыба": {"cal": 120, "prot": 20, "fat": 5, "carb": 0, "default_g": 150},
    "лосось": {"cal": 208, "prot": 20, "fat": 13, "carb": 0, "default_g": 150},
    "тунец": {"cal": 132, "prot": 28, "fat": 1, "carb": 0, "default_g": 150},
    "яйцо": {"cal": 70, "prot": 6, "fat": 5, "carb": 1, "default_g": 60},
    "яйца": {"cal": 140, "prot": 12, "fat": 10, "carb": 2, "default_g": 120},
    "творог": {"cal": 100, "prot": 15, "fat": 4, "carb": 3, "default_g": 150},
    "сыр": {"cal": 300, "prot": 20, "fat": 25, "carb": 2, "default_g": 30},
    "молоко": {"cal": 50, "prot": 3, "fat": 2.5, "carb": 5, "default_g": 200},
    "кефир": {"cal": 40, "prot": 3, "fat": 1, "carb": 4, "default_g": 200},
    "йогурт": {"cal": 60, "prot": 5, "fat": 2, "carb": 6, "default_g": 150},
    
    # Углеводы
    "гречка": {"cal": 110, "prot": 4, "fat": 1, "carb": 21, "default_g": 150},
    "рис": {"cal": 130, "prot": 2.7, "fat": 0.3, "carb": 28, "default_g": 150},
    "овсянка": {"cal": 150, "prot": 5, "fat": 2, "carb": 28, "default_g": 150},
    "макароны": {"cal": 158, "prot": 6, "fat": 0.9, "carb": 31, "default_g": 150},
    "паста": {"cal": 158, "prot": 6, "fat": 0.9, "carb": 31, "default_g": 150},
    "картофель": {"cal": 77, "prot": 2, "fat": 0.1, "carb": 17, "default_g": 150},
    "хлеб": {"cal": 265, "prot": 9, "fat": 3, "carb": 49, "default_g": 30},
    "тост": {"cal": 265, "prot": 9, "fat": 3, "carb": 49, "default_g": 30},
    "банан": {"cal": 90, "prot": 1, "fat": 0.3, "carb": 23, "default_g": 120},
    "яблоко": {"cal": 52, "prot": 0.3, "fat": 0.2, "carb": 14, "default_g": 180},

    # Овощи
    "салат": {"cal": 30, "prot": 1, "fat": 0.3, "carb": 5, "default_g": 150},
    "овощи": {"cal": 40, "prot": 2, "fat": 0.5, "carb": 7, "default_g": 200},
    "брокколи": {"cal": 34, "prot": 2.8, "fat": 0.4, "carb": 7, "default_g": 150},
    "помидор": {"cal": 18, "prot": 0.9, "fat": 0.2, "carb": 4, "default_g": 150},
    
    # Фастфуд / Сладкое
    "пицца": {"cal": 285, "prot": 12, "fat": 10, "carb": 36, "default_g": 250},
    "бургер": {"cal": 295, "prot": 17, "fat": 14, "carb": 24, "default_g": 200},
    "торт": {"cal": 350, "prot": 5, "fat": 15, "carb": 50, "default_g": 100},
    "шоколад": {"cal": 550, "prot": 5, "fat": 32, "carb": 60, "default_g": 30},

    # C37
    "батончик": {"cal": 425, "prot": 32, "fat": 18, "carb": 30, "default_g": 40},
    "c37": {"cal": 425, "prot": 32, "fat": 18, "carb": 30, "default_g": 40},

    # Напитки
    "кофе": {"cal": 5, "prot": 0.3, "fat": 0, "carb": 1, "default_g": 200},
    "чай": {"cal": 1, "prot": 0, "fat": 0, "carb": 0, "default_g": 200},
}

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
            temperature=0.1, 
            verify_ssl_certs=False
        )
    return _llm

def get_retriever():
    global _retriever
    if _retriever is None:
        logger.info("Загрузка базы знаний (BM25)...")
        if not KNOWLEDGE_DIR.exists():
            KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)
        loader = DirectoryLoader(str(KNOWLEDGE_DIR), glob="**/*.md", loader_cls=TextLoader, loader_kwargs={"autodetect_encoding": True})
        docs = loader.load()
        if docs:
            splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
            chunks = splitter.split_documents(docs)
            _retriever = BM25Retriever.from_documents(chunks)
            _retriever.k = 3
    return _retriever

@app.on_event("startup")
async def startup_event():
    get_llm()
    get_retriever()

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
            context = "(Ошибка поиска)"
            docs = []
    
    prompt = ChatPromptTemplate.from_messages([("system", SYSTEM_PROMPT), ("human", "{question}")])
    messages = prompt.format_messages(context=context, question=req.message)
    
    try:
        llm = get_llm()
        response = llm.invoke(messages)
        reply = response.content
    except Exception as e:
        logger.error(f"Ошибка GigaChat (чат): {e}")
        raise HTTPException(status_code=502, detail="Ошибка генерации ответа")
        
    sources = [SourceInfo(source="knowledge", snippet=d.page_content[:100]) for d in docs]
    return ChatResponse(reply=reply, sources=sources)


# === НОВЫЙ ПАРСИНГ ЕДЫ (ГИГАЧАТ ИЗВЛЕКАЕТ, ПИТОН СЧИТАЕТ) ===
@app.post("/api/parse-food")
async def parse_food(req: FoodParseRequest):
    logger.info(f"Запрос на парсинг еды: {req.text}")
    
    parsed_items = []
    
    # 1. ПРОСИМ GIGACHAT ИЗВЛЕЧЬ ТОЛЬКО НАЗВАНИЯ И ВЕС
    try:
        llm = get_llm()
        # Даем список продуктов, чтобы GigaChat знал словарь
        available = ", ".join(FOOD_DB.keys())
        
        messages = [
            ("system", f"""Ты — экстрактор данных из текста о еде.
Твоя задача: найти в тексте продукты и их вес в граммах. НИКАКИХ расчетов калорий!
Доступные продукты: {available}
Если пользователь не указал вес — оставь null.

Верни СТРОГО JSON-массив. Пример:
[{{"name": "курица", "grams": 270}}, {{"name": "гречка", "grams": 150}}]

Если еды нет — верни []. Никакого текста до или после."""),
            ("human", f"Текст: {req.text}")
        ]
        
        response = llm.invoke(messages)
        content = response.content.strip()
        logger.info(f"GigaChat извлек: {content}")
        
        start = content.find('[')
        end = content.rfind(']')
        if start != -1 and end != -1:
            parsed_items = json.loads(content[start:end+1])
            
    except Exception as e:
        logger.error(f"GigaChat не справился: {e}")

    # 2. ЕСЛИ GIGACHAT УПАЛ, ИСПОЛЬЗУЕМ ЛОКАЛЬНЫЙ ЭКСТРАКТОР
    if not parsed_items:
        logger.info("Использую локальный экстрактор.")
        text_lower = req.text.lower()
        for key in FOOD_DB:
            if key in text_lower:
                idx = text_lower.find(key)
                context = text_lower[max(0, idx-20):idx+len(key)+20]
                grams_match = re.search(r'(\d+)\s*(?:г|грамм|гр)', context)
                grams = int(grams_match.group(1)) if grams_match else None
                parsed_items.append({"name": key, "grams": grams})

    if not parsed_items:
        return {"items": [], "error": "Не удалось найти еду в тексте"}

    # 3. СЧИТАЕМ КАЛОРИИ И БЖУ ПО НАШЕЙ БАЗЕ
    result_items = []
    for item in parsed_items:
        name = item.get("name", "").lower().strip()
        grams = item.get("grams")
        
        found_key = None
        for key in FOOD_DB:
            if key in name or name in key:
                found_key = key
                break
        
        if not found_key:
            continue
        
        product = FOOD_DB[found_key]
        weight = grams if grams else product["default_g"]
        
        factor = weight / 100
        calc_item = {
            "name": found_key.capitalize(),
            "grams": round(weight),
            "cal": round(product["cal"] * factor),
            "prot": round(product["prot"] * factor, 1),
            "fat": round(product["fat"] * factor, 1),
            "carb": round(product["carb"] * factor, 1),
        }
        result_items.append(calc_item)
    
    if not result_items:
        return {"items": [], "error": "Продукты не распознаны в базе"}
    
    logger.info(f"Итог: {result_items}")
    return {"items": result_items}
