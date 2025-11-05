
from dotenv import load_dotenv
from openai import OpenAI, APIError, APITimeoutError, APIConnectionError, APIStatusError
import os

load_dotenv()
LLM_API_KEY = os.getenv("LLM_API_KEY")
EMBEDDER_API_KEY = os.getenv("EMBEDDER_API_KEY")

# --- Названия моделей ---
EMBEDDER_MODEL = 'text-embedding-3-small'
LLM_MODEL_NAME = "openrouter/meta-llama/llama-3-70b-instruc"

# --- Инициализация клиентов OpenAI ---
# Клиент для LLM
client_llm = None
if LLM_API_KEY:
    client_llm = OpenAI(
        base_url="https://ai-for-finance-hack.up.railway.app/",
        api_key=LLM_API_KEY,
    )
    print(f"Клиент LLM инициализирован. Ключ: ...{LLM_API_KEY[-4:]}")
else:
    print("Ключ LLM_API_KEY не найден. Функции, использующие LLM, могут не работать.")

# Клиент для эмбеддингов
client_embedder = None
if EMBEDDER_API_KEY:
    client_embedder = OpenAI(
        base_url="https://ai-for-finance-hack.up.railway.app/",
        api_key=EMBEDDER_API_KEY,
    )
    print(f"Клиент для эмбеддингов инициализирован. Ключ: ...{EMBEDDER_API_KEY[-4:]}")
else:
    print("Ключ EMBEDDER_API_KEY не найден. Функции, использующие эмбеддинги, могут не работать.")
