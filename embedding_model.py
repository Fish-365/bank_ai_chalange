import time
import random
from openai import OpenAI, APIError, APITimeoutError, APIConnectionError, APIStatusError
from typing import Optional
from dotenv import load_dotenv
import os

# --- Настройка клиента ---
load_dotenv()
EMBEDDER_API_KEY = os.getenv("EMBEDDER_API_KEY")

print(f"Используемый API ключ для эмбеддингов: {EMBEDDER_API_KEY[-4:] if EMBEDDER_API_KEY else 'Не найден'}")

# Указывайте модель без префикса 'openai/' для стандартного API
MODEL = 'text-embedding-3-small'

# Создаем клиент один раз на уровне модуля для переиспользования
client = OpenAI(
    base_url="https://ai-for-finance-hack.up.railway.app/",
    api_key=EMBEDDER_API_KEY,
)

def get_embedding(text: str, max_retries: int = 5, initial_backoff: int = 1) -> Optional[list[float]]:
    """
    Получает векторное представление (эмбеддинг) для текста.
    Использует механизм повторных запросов с экспоненциальной задержкой.

    :param text: Текст для получения эмбеддинга.
    :param max_retries: Максимальное количество попыток.
    :param initial_backoff: Начальная задержка между попытками в секундах.
    :return: Вектор эмбеддинга или None в случае неудачи.
    """
    backoff_time = initial_backoff
    
    for attempt in range(max_retries):
        try:
            response = client.embeddings.create(
                model=MODEL,
                input=text
            )
            return response.data[0].embedding

        except (APITimeoutError, APIConnectionError, APIStatusError) as e:
            is_retriable = isinstance(e, (APITimeoutError, APIConnectionError)) or \
                           (isinstance(e, APIStatusError) and e.status_code >= 500)

            if is_retriable and attempt < max_retries - 1:
                print(f"Ошибка API эмбеддингов (попытка {attempt + 1}/{max_retries}): {e}. Повтор через {backoff_time:.2f} сек.")
                time.sleep(backoff_time)
                backoff_time = (backoff_time * 2) + random.uniform(0, 1)
            else:
                print(f"Критическая ошибка API эмбеддингов после {attempt + 1} попыток: {e}")
                return None

        except APIError as e:
            print(f"Неустранимая ошибка API эмбеддингов: {e}")
            return None

    print("Не удалось получить эмбеддинг после всех попыток.")
    return None