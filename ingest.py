"""
Собрать индекс FAISS офлайн: python ingest.py [--force]

Результат: папка faiss_index/ и файл knowledge_cache.json.
Их можно закоммитить / положить в образ — тогда холодный старт
на Netrun не будет вызывать GigaChat Embeddings.
"""
import sys
import asyncio
import logging

import app as rag

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ingest")


async def main():
    force = "--force" in sys.argv

    if force:
        logger.info("Удаляю старый кэш эмбеддингов и индекс FAISS")
        if rag.EMBEDDINGS_CACHE.exists():
            rag.EMBEDDINGS_CACHE.unlink()
        if rag.FAISS_INDEX_DIR.exists():
            import shutil
            shutil.rmtree(rag.FAISS_INDEX_DIR)

    await rag._ensure_initialized()
    count = rag._vectorstore.index.ntotal
    logger.info(f"Готово. Векторов в FAISS: {count}")


if __name__ == "__main__":
    asyncio.run(main())
