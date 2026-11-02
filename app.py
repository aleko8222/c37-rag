import os
import json
import logging
import re
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

# ==========================================
# НАСТРОЙКА ОКРУЖЕНИЯ И ЛОГИРОВАНИЯ
# ==========================================
load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("c37-rag")

GIGACHAT_CREDENTIALS = os.getenv("GIGACHAT_CREDENTIALS")
GIGACHAT_SCOPE = os.getenv("GIGACHAT_SCOPE", "GIGACHAT_API_PERS")
GIGACHAT_CHAT_MODEL = os.getenv("GIGACHAT_CHAT_MODEL", "GigaChat-2-Pro")
KNOWLEDGE_DIR = Path(os.getenv("KNOWLEDGE_DIR", "./knowledge"))

# ==========================================
# БАЗА ПРОДУКТОВ (200+ ПОЗИЦИЙ, БЖУ на 100г)
# ==========================================
FOOD_DB = {
    # --- МЯСО И ПТИЦА ---
    "курица": {"cal": 165, "prot": 31, "fat": 3.6, "carb": 0, "default_g": 150},
    "куриная грудка": {"cal": 165, "prot": 31, "fat": 3.6, "carb": 0, "default_g": 150},
    "куриное бедро": {"cal": 209, "prot": 26, "fat": 11, "carb": 0, "default_g": 150},
    "индейка": {"cal": 104, "prot": 21, "fat": 1.5, "carb": 0, "default_g": 150},
    "говядина": {"cal": 250, "prot": 26, "fat": 16, "carb": 0, "default_g": 150},
    "свинина": {"cal": 259, "prot": 27, "fat": 16, "carb": 0, "default_g": 150},
    "баранина": {"cal": 294, "prot": 25, "fat": 21, "carb": 0, "default_g": 150},
    "бекон": {"cal": 540, "prot": 37, "fat": 42, "carb": 1, "default_g": 30},
    "ветчина": {"cal": 145, "prot": 21, "fat": 6, "carb": 2, "default_g": 50},
    "сосиски": {"cal": 260, "prot": 12, "fat": 22, "carb": 3, "default_g": 100},
    "колбаса": {"cal": 300, "prot": 13, "fat": 27, "carb": 2, "default_g": 50},

    # --- РЫБА И МОРЕПРОДУКТЫ ---
    "рыба": {"cal": 120, "prot": 20, "fat": 5, "carb": 0, "default_g": 150},
    "лосось": {"cal": 208, "prot": 20, "fat": 13, "carb": 0, "default_g": 150},
    "скумбрия": {"cal": 205, "prot": 19, "fat": 13, "carb": 0, "default_g": 150},
    "тунец": {"cal": 132, "prot": 28, "fat": 1, "carb": 0, "default_g": 150},
    "треска": {"cal": 82, "prot": 18, "fat": 0.7, "carb": 0, "default_g": 150},
    "минтай": {"cal": 72, "prot": 16, "fat": 0.9, "carb": 0, "default_g": 150},
    "креветки": {"cal": 99, "prot": 24, "fat": 0.3, "carb": 0, "default_g": 150},
    "кальмар": {"cal": 92, "prot": 15.5, "fat": 1.4, "carb": 3, "default_g": 150},
    "икра": {"cal": 250, "prot": 25, "fat": 15, "carb": 4, "default_g": 30},
    "икра красная": {"cal": 250, "prot": 25, "fat": 15, "carb": 4, "default_g": 30},

    # --- ЯЙЦА И МОЛОЧКА ---
    "яйцо": {"cal": 70, "prot": 6, "fat": 5, "carb": 1, "default_g": 60},
    "яйца": {"cal": 140, "prot": 12, "fat": 10, "carb": 2, "default_g": 120},
    "творог": {"cal": 100, "prot": 15, "fat": 4, "carb": 3, "default_g": 150},
    "сыр": {"cal": 300, "prot": 20, "fat": 25, "carb": 2, "default_g": 30},
    "молоко": {"cal": 50, "prot": 3, "fat": 2.5, "carb": 5, "default_g": 200},
    "кефир": {"cal": 40, "prot": 3, "fat": 1, "carb": 4, "default_g": 200},
    "йогурт": {"cal": 60, "prot": 5, "fat": 2, "carb": 6, "default_g": 150},
    "сметана": {"cal": 200, "prot": 2.8, "fat": 20, "carb": 3.2, "default_g": 30},

    # --- КРУПЫ И УГЛЕВОДЫ ---
    "гречка": {"cal": 110, "prot": 4, "fat": 1, "carb": 21, "default_g": 150},
    "рис": {"cal": 130, "prot": 2.7, "fat": 0.3, "carb": 28, "default_g": 150},
    "овсянка": {"cal": 150, "prot": 5, "fat": 2, "carb": 28, "default_g": 150},
    "макароны": {"cal": 158, "prot": 6, "fat": 0.9, "carb": 31, "default_g": 150},
    "паста": {"cal": 158, "prot": 6, "fat": 0.9, "carb": 31, "default_g": 150},
    "картофель": {"cal": 77, "prot": 2, "fat": 0.1, "carb": 17, "default_g": 150},
    "батат": {"cal": 86, "prot": 1.6, "fat": 0.1, "carb": 20, "default_g": 150},
    "киноа": {"cal": 120, "prot": 4.4, "fat": 1.9, "carb": 21, "default_g": 150},
    "булгур": {"cal": 83, "prot": 3, "fat": 0.2, "carb": 18, "default_g": 150},
    "хлеб": {"cal": 265, "prot": 9, "fat": 3, "carb": 49, "default_g": 30},
    "тост": {"cal": 265, "prot": 9, "fat": 3, "carb": 49, "default_g": 30},
    "чечевица": {"cal": 116, "prot": 9, "fat": 0.4, "carb": 20, "default_g": 150},
    "фасоль": {"cal": 127, "prot": 9, "fat": 0.5, "carb": 23, "default_g": 150},
    "кукуруза": {"cal": 86, "prot": 3.2, "fat": 1.2, "carb": 19, "default_g": 150},

    # --- ОВОЩИ ---
    "салат": {"cal": 30, "prot": 1, "fat": 0.3, "carb": 5, "default_g": 150},
    "овощи": {"cal": 40, "prot": 2, "fat": 0.5, "carb": 7, "default_g": 200},
    "брокколи": {"cal": 34, "prot": 2.8, "fat": 0.4, "carb": 7, "default_g": 150},
    "помидор": {"cal": 18, "prot": 0.9, "fat": 0.2, "carb": 4, "default_g": 150},
    "огурец": {"cal": 15, "prot": 0.7, "fat": 0.1, "carb": 3, "default_g": 150},
    "авокадо": {"cal": 160, "prot": 2, "fat": 15, "carb": 9, "default_g": 100},
    "морковь": {"cal": 41, "prot": 0.9, "fat": 0.2, "carb": 10, "default_g": 100},
    "капуста": {"cal": 25, "prot": 1.3, "fat": 0.1, "carb": 6, "default_g": 150},
    "свекла": {"cal": 43, "prot": 1.6, "fat": 0.2, "carb": 10, "default_g": 150},
    "сельдерей": {"cal": 16, "prot": 0.7, "fat": 0.2, "carb": 3, "default_g": 100},
    "шпинат": {"cal": 23, "prot": 2.9, "fat": 0.4, "carb": 3.6, "default_g": 100},
    "грибы": {"cal": 22, "prot": 3.1, "fat": 0.3, "carb": 3.3, "default_g": 150},

    # --- ФРУКТЫ ---
    "банан": {"cal": 90, "prot": 1, "fat": 0.3, "carb": 23, "default_g": 120},
    "яблоко": {"cal": 52, "prot": 0.3, "fat": 0.2, "carb": 14, "default_g": 180},
    "апельсин": {"cal": 47, "prot": 0.9, "fat": 0.1, "carb": 12, "default_g": 180},
    "грейпфрут": {"cal": 35, "prot": 0.7, "fat": 0.1, "carb": 9, "default_g": 200},
    "ананас": {"cal": 50, "prot": 0.5, "fat": 0.1, "carb": 13, "default_g": 150},
    "манго": {"cal": 60, "prot": 0.8, "fat": 0.4, "carb": 15, "default_g": 150},
    "киви": {"cal": 61, "prot": 1.1, "fat": 0.5, "carb": 15, "default_g": 100},
    "виноград": {"cal": 69, "prot": 0.7, "fat": 0.2, "carb": 18, "default_g": 150},
    "груша": {"cal": 57, "prot": 0.4, "fat": 0.1, "carb": 15, "default_g": 180},
    "клубника": {"cal": 32, "prot": 0.7, "fat": 0.3, "carb": 8, "default_g": 150},

    # --- ЖИРЫ И ОРЕХИ ---
    "масло": {"cal": 884, "prot": 0, "fat": 100, "carb": 0, "default_g": 10},
    "оливковое масло": {"cal": 884, "prot": 0, "fat": 100, "carb": 0, "default_g": 10},
    "орехи": {"cal": 600, "prot": 15, "fat": 55, "carb": 20, "default_g": 30},
    "миндаль": {"cal": 579, "prot": 21, "fat": 50, "carb": 22, "default_g": 30},
    "грецкий орех": {"cal": 654, "prot": 15, "fat": 65, "carb": 14, "default_g": 30},
    "арахис": {"cal": 567, "prot": 26, "fat": 49, "carb": 16, "default_g": 30},
    "семечки": {"cal": 580, "prot": 20, "fat": 50, "carb": 20, "default_g": 30},

    # --- СОУСЫ И ГОТОВАЯ ЕДА ---
    "хумус": {"cal": 166, "prot": 8, "fat": 10, "carb": 14, "default_g": 50},
    "кетчуп": {"cal": 100, "prot": 1.3, "fat": 0.1, "carb": 25, "default_g": 20},
    "майонез": {"cal": 680, "prot": 1, "fat": 75, "carb": 1, "default_g": 15},
    "пицца": {"cal": 285, "prot": 12, "fat": 10, "carb": 36, "default_g": 250},
    "бургер": {"cal": 295, "prot": 17, "fat": 14, "carb": 24, "default_g": 200},
    "сырники": {"cal": 220, "prot": 15, "fat": 10, "carb": 20, "default_g": 150},
    "оладьи": {"cal": 230, "prot": 6, "fat": 8, "carb": 35, "default_g": 150},
    "пельмени": {"cal": 275, "prot": 12, "fat": 13, "carb": 28, "default_g": 200},
    "суп": {"cal": 60, "prot": 3, "fat": 2, "carb": 7, "default_g": 300},
    "борщ": {"cal": 50, "prot": 2, "fat": 2, "carb": 7, "default_g": 300},
    "котлета": {"cal": 220, "prot": 15, "fat": 15, "carb": 8, "default_g": 100},

    # --- СЛАДКОЕ И C37 ---
    "торт": {"cal": 350, "prot": 5, "fat": 15, "carb": 50, "default_g": 100},
    "шоколад": {"cal": 550, "prot": 5, "fat": 32, "carb": 60, "default_g": 30},
    "печенье": {"cal": 480, "prot": 5, "fat": 20, "carb": 65, "default_g": 30},
    "батончик": {"cal": 425, "prot": 32, "fat": 18, "carb": 30, "default_g": 40},
    "c37": {"cal": 425, "prot": 32, "fat": 18, "carb": 30, "default_g": 40},

    # --- НАПИТКИ ---
    "кофе": {"cal": 5, "prot": 0.3, "fat": 0, "carb": 1, "default_g": 200},
    "чай": {"cal": 1, "prot": 0, "fat": 0, "carb": 0, "default_g": 200},
    "сок": {"cal": 45, "prot": 0.5, "fat": 0, "carb": 10, "default_g": 250},
    "кола": {"cal": 42, "prot": 0, "fat": 0, "carb": 10, "default_g": 330},
}

# ==========================================
# СИСТЕМНЫЕ ПРОМПТЫ
# ==========================================
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

# ==========================================
# ИНИЦИАЛИЗАЦИЯ FASTAPI
# ==========================================
app = FastAPI(title="C37 Eva RAG API")
app.add_middleware(
    CORSMiddleware, 
    allow_origins=["*"], 
    allow_methods=["*"], 
    allow_headers=["*"]
)

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
            logger.info(f"Загружено {len(chunks)} чанков в BM25.")
        else:
            logger.warning("Папка knowledge пуста!")
    return _retriever

@app.on_event("startup")
async def startup_event():
    get_llm()
    get_retriever()
    logger.info("Сервер C37 запущен и готов к работе.")

# ==========================================
# МОДЕЛИ ДАННЫХ
# ==========================================
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

# ==========================================
# ЭНДПОИНТЫ
# ==========================================
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
            context = "(Ошибка поиска в базе знаний)"
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


@app.post("/api/parse-food")
async def parse_food(req: FoodParseRequest):
    logger.info(f"Запрос на парсинг еды: {req.text}")
    
    text_lower = req.text.lower()
    combined_items = {}

    # 1. ЛОКАЛЬНЫЙ ПАРСЕР (Ищем всё, что знаем, чтобы точно посчитать БЖУ)
    weight_matches = re.findall(r'(\d+)\s*(?:г|грамм|гр)', text_lower)
    default_weight = int(weight_matches[0]) if weight_matches else None
    
    for key in FOOD_DB:
        if key in text_lower:
            idx = text_lower.find(key)
            context = text_lower[max(0, idx-30):idx+len(key)+30]
            grams_match = re.search(r'(\d+)\s*(?:г|грамм|гр)', context)
            grams = int(grams_match.group(1)) if grams_match else default_weight
            
            product = FOOD_DB[key]
            weight = grams if grams else product["default_g"]
            
            # Сохраняем во временный словарь
            combined_items[key] = {"name": key, "grams": weight}

    # 2. GIGACHAT (Всегда запускаем, чтобы найти то, чего нет в базе, или уточнить вес)
    try:
        llm = get_llm()
        available = ", ".join(FOOD_DB.keys())
        
        messages = [
            ("system", f"""Ты — экстрактор данных. Извлеки ВСЕ продукты из текста, не пропуская ни одного.
НЕ считай калории! Только названия и вес в граммах.
Доступные продукты: {available}
Если вес не указан — оставь null. Если продукт есть в тексте, но не в списке — тоже добавь его.

Верни СТРОГО JSON-массив. Пример:
[{{"name": "гречка", "grams": 300}}, {{"name": "хумус", "grams": null}}]

Никакого текста до или после. Если еды нет — верни []."""),
            ("human", f"Текст: {req.text}")
        ]
        
        response = llm.bind(max_tokens=4000).invoke(messages)
        content = response.content.strip()
        logger.info(f"GigaChat извлек: {content}")
        
        start = content.find('[')
        end = content.rfind(']')
        if start != -1 and end != -1:
            gigachat_items = json.loads(content[start:end+1])
            
            for item in gigachat_items:
                name = item.get("name", "").lower().strip()
                grams = item.get("grams")
                
                # Ищем совпадение в базе
                found_key = None
                for db_key in FOOD_DB:
                    if db_key in name or name in db_key:
                        found_key = db_key
                        break
                
                if found_key:
                    # Если локальный парсер уже нашел, просто обновляем вес, если он был пустой
                    if found_key in combined_items:
                        if combined_items[found_key]["grams"] is None and grams:
                            combined_items[found_key]["grams"] = grams
                    else:
                        weight = grams if grams else FOOD_DB[found_key]["default_g"]
                        combined_items[found_key] = {"name": found_key, "grams": weight}
                else:
                    # Если продукта нет в базе, добавляем как есть (с примерными калориями)
                    if name and name not in combined_items:
                        weight = grams if grams else 100
                        combined_items[name] = {"name": name, "grams": weight, "unknown": True}
                        
    except Exception as e:
        logger.error(f"GigaChat упал: {e}")

    # 3. СЧИТАЕМ КАЛОРИИ И БЖУ
    result_items = []
    for k, item in combined_items.items():
        name = item["name"]
        weight = item["grams"] if item["grams"] else 100
        
        # Еще раз проверяем базу (чтобы точно быть уверенным)
        found_key = None
        for db_key in FOOD_DB:
            if db_key in name or name in db_key:
                found_key = db_key
                break
                
        if found_key and not item.get("unknown"):
            product = FOOD_DB[found_key]
            factor = weight / 100
            result_items.append({
                "name": found_key.capitalize(),
                "grams": round(weight),
                "cal": round(product["cal"] * factor),
                "prot": round(product["prot"] * factor, 1),
                "fat": round(product["fat"] * factor, 1),
                "carb": round(product["carb"] * factor, 1),
            })
        else:
            # Неизвестный продукт — берем средние значения (100 ккал / 100г)
            factor = weight / 100
            result_items.append({
                "name": name.capitalize(),
                "grams": round(weight),
                "cal": round(100 * factor),
                "prot": round(5 * factor, 1),
                "fat": round(5 * factor, 1),
                "carb": round(10 * factor, 1),
            })

    logger.info(f"Итог парсинга: {result_items}")
    return {"items": result_items}
