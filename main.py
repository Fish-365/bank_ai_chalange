# -*- coding: utf-8 -*"
import os
import re
import sys
import time
import random
import asyncio
from typing import Optional, Tuple, List, Dict, Any, Set, NamedTuple

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI, APIError, APITimeoutError, APIConnectionError, APIStatusError, AsyncOpenAI
from tqdm import tqdm

# --- Глобальные переменные и константы ---

# --- Пути к файлам ---
TRAIN_DATA_PATH = 'baseline/train_data.csv'
QUESTIONS_PATH = 'baseline/questions.csv'
SUBMISSION_PATH = 'submission.csv'

# --- Параметры для API ---
MAX_API_RETRIES = 5
INITIAL_API_BACKOFF = 1

# --- Загрузка переменных окружения ---
load_dotenv()
LLM_API_KEY = os.getenv("LLM_API_KEY")
EMBEDDER_API_KEY = os.getenv("EMBEDDER_API_KEY")

# --- Названия моделей ---
EMBEDDER_MODEL = 'text-embedding-3-small'
LLM_MODEL_NAME = "openrouter/mistralai/mistral-small-3.2-24b-instruct"

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

# Асинхронный клиент для эмбеддингов
async_client_embedder = None
if EMBEDDER_API_KEY:
    async_client_embedder = AsyncOpenAI(
        base_url="https://ai-for-finance-hack.up.railway.app/",
        api_key=EMBEDDER_API_KEY,
    )
    print(f"Асинхронный клиент для эмбеддингов инициализирован. Ключ: ...{EMBEDDER_API_KEY[-4:]}")
else:
    print("Ключ EMBEDDER_API_KEY не найден. Функции, использующие эмбеддинги, могут не работать.")


# --- Типы данных для улучшения читаемости ---
class KnowledgeBase(NamedTuple):
    docs_df: pd.DataFrame
    chunks_df: pd.DataFrame

# --- Логика из embedding_model.py ---

async def get_embedding_async(text: str, max_retries: int = MAX_API_RETRIES, initial_backoff: int = INITIAL_API_BACKOFF) -> Optional[list[float]]:
    """
    Асинхронно получает векторное представление (эмбеддинг) для текста.
    Использует механизм повторных запросов с экспоненциальной задержкой.
    """
    if not async_client_embedder:
        print("Асинхронный клиент для эмбеддингов не инициализирован.")
        return None

    backoff_time = initial_backoff
    for attempt in range(max_retries):
        try:
            response = await async_client_embedder.embeddings.create(
                model=EMBEDDER_MODEL,
                input=text
            )
            return response.data[0].embedding
        except (APITimeoutError, APIConnectionError, APIStatusError) as e:
            is_retriable = isinstance(e, (APITimeoutError, APIConnectionError)) or \
                           (isinstance(e, APIStatusError) and e.status_code >= 500)
            if is_retriable and attempt < max_retries - 1:
                print(f"Ошибка API эмбеддингов (попытка {attempt + 1}/{max_retries}): {e}. Повтор через {backoff_time:.2f} сек.")
                await asyncio.sleep(backoff_time)
                backoff_time = (backoff_time * 2) + random.uniform(0, 1)
            else:
                print(f"Критическая ошибка API эмбеддингов после {attempt + 1} попыток: {e}")
                return None
        except APIError as e:
            print(f"Неустранимая ошибка API эмбеддингов: {e}")
            return None
    print("Не удалось получить эмбеддинг после всех попыток.")
    return None

# --- Логика из read_base.py ---

def clean_text(text: str) -> str:
    """Очищает текст: lowercase, удаляет лишние пробелы и неалфавитные символы."""
    text = text.lower()
    text = re.sub(r'[^a-zа-яё0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def split_long_paragraph(paragraph: str, max_length: int = 500) -> list[str]:
    """Разбивает слишком длинный параграф на части, пригодные для эмбеддинга."""
    if len(paragraph) <= max_length:
        return [paragraph]
    sentences = re.split(r'(?<=[.!?])\s+', paragraph)
    if all(len(s) <= max_length for s in sentences):
        return [s for s in sentences if s]
    words = paragraph.split()
    chunks = []
    current_chunk = ""
    for word in words:
        if len(current_chunk) + len(word) + 1 <= max_length:
            current_chunk += " " + word if current_chunk else word
        else:
            chunks.append(current_chunk)
            current_chunk = word
    if current_chunk:
        chunks.append(current_chunk)
    return chunks

def get_all_tags(docs_df: pd.DataFrame) -> Tuple[str, ...]:
    """Собирает все уникальные теги из DataFrame документов."""
    if 'tags' not in docs_df.columns:
        return tuple()
    all_tags: Set[str] = set()
    for tags_str in docs_df['tags'].dropna():
        all_tags.update(tag.strip() for tag in tags_str.split(','))
    return tuple(sorted(list(all_tags)))

CONCURRENT_REQUEST_LIMIT = 120

async def get_embedding_with_item(item: Dict[str, Any], semaphore: asyncio.Semaphore) -> Tuple[Dict[str, Any], Optional[list[float]]]:
    """Получает эмбеддинг для элемента и возвращает его вместе с элементом, используя семафор."""
    async with semaphore:
        embedding = await get_embedding_async(item['cleaned'])
        return item, embedding

async def create_rag_database_async(file_path: str = TRAIN_DATA_PATH, row_limit: Optional[int] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Асинхронно создает базу данных для RAG из CSV-файла с отображением прогресса и ограничением конкурентности."""
    pattern = r'Обновлено \d{2}\.\d{2}\.\d{4} в \d{1,2}:\d{2}'
    question_pattern = r"^##.*"
    empty_dfs = pd.DataFrame(), pd.DataFrame()
    try:
        df = pd.read_csv(file_path)
        if row_limit:
            df = df.head(row_limit)
    except FileNotFoundError:
        print(f"Ошибка: Файл '{file_path}' не найден.")
        return empty_dfs

    documents_data = []
    items_to_process = []
    print("Подготовка данных для обработки...")
    for index, row in df.iterrows():
        document_text = str(row.get('text', ''))
        tag = str(row.get('tags', ''))
        cleaned_document = re.sub(pattern, '', document_text).strip()
        doc_id = index
        documents_data.append({'doc_id': doc_id, 'original_text': document_text, 'tags': tag})
        paragraphs = [p.strip() for p in cleaned_document.split('\n\n') if p.strip()]
        for para in paragraphs:
            if re.match(question_pattern, para):
                continue
            sub_chunks = split_long_paragraph(para, max_length=500)
            for chunk in sub_chunks:
                cleaned = clean_text(chunk)
                if cleaned:
                    items_to_process.append({'doc_id': doc_id, 'original': chunk, 'cleaned': cleaned})

    semaphore = asyncio.Semaphore(CONCURRENT_REQUEST_LIMIT)
    tasks = [get_embedding_with_item(item, semaphore) for item in items_to_process]
    
    final_result = []
    
    print(f"Запускается асинхронное получение {len(tasks)} эмбеддингов с ограничением в {CONCURRENT_REQUEST_LIMIT} одновременных запросов...")
    
    for future in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Получение эмбеддингов"):
        item, embedding = await future
        if embedding is not None:
            final_result.append((item['doc_id'], item['original'], item['cleaned'], embedding))

    if not final_result:
        print("\nЭмбеддинги не были созданы. Возвращается пустая база данных.")
        return empty_dfs

    docs_df = pd.DataFrame(documents_data)
    chunks_df = pd.DataFrame(final_result, columns=['doc_id', 'original_chunk', 'cleaned_chunk', 'embedding'])
    print(f"\nОбработка завершена. Создано {len(docs_df)} документов и {len(chunks_df)} чанков с эмбеддингами.")
    return docs_df, chunks_df

async def find_relevant_docs_async(question: str, selected_tags: tuple, docs_df: pd.DataFrame, chunks_df: pd.DataFrame, top_k: int = 5, vector_weight: float = 0.7, tag_weight: float = 0.3) -> List[Dict[str, Any]]:
    """Находит наиболее релевантные документы, используя гибридный поиск."""
    if chunks_df.empty:
        return []
    question_embedding = await get_embedding_async(question)
    if question_embedding is None:
        print("Не удалось получить эмбеддинг для вопроса.")
        return []

    chunk_embeddings = np.array(chunks_df['embedding'].tolist())
    question_embedding_np = np.array(question_embedding)
    
    similarities = np.dot(chunk_embeddings, question_embedding_np) / (np.linalg.norm(chunk_embeddings, axis=1) * np.linalg.norm(question_embedding_np))
    chunks_df['vector_score'] = similarities

    merged_df = pd.merge(chunks_df, docs_df, on='doc_id')
    
    def calculate_tag_score(row):
        doc_tags = set(str(row['tags']).split(','))
        matched_tags = doc_tags.intersection(selected_tags)
        return len(matched_tags)

    merged_df['tag_score'] = merged_df.apply(calculate_tag_score, axis=1)

    vec_min, vec_max = merged_df['vector_score'].min(), merged_df['vector_score'].max()
    tag_min, tag_max = merged_df['tag_score'].min(), merged_df['tag_score'].max()

    if (vec_max - vec_min) > 1e-9:
        merged_df['norm_vector_score'] = (merged_df['vector_score'] - vec_min) / (vec_max - vec_min)
    else:
        merged_df['norm_vector_score'] = 0.5 if not merged_df.empty else 0.0

    if (tag_max - tag_min) > 0:
        merged_df['norm_tag_score'] = (merged_df['tag_score'] - tag_min) / (tag_max - tag_min)
    else:
        merged_df['norm_tag_score'] = 0.5 if not merged_df.empty and tag_max > 0 else 0.0

    merged_df['final_score'] = (merged_df['norm_vector_score'] * vector_weight) + (merged_df['norm_tag_score'] * tag_weight)

    top_docs = merged_df.loc[merged_df.groupby('doc_id')['final_score'].idxmax()]
    top_docs = top_docs.sort_values(by='final_score', ascending=False).head(top_k)

    result = []
    for _, row in top_docs.iterrows():
        result.append({
            'text': row['original_text'],
            'score': row['final_score']
        })
    return result

# --- Логика из model.py ---

def choose_tags_LLM(all_tags: Tuple[str, ...], question: str) -> Tuple[str, ...]:
    """Выбирает релевантные теги для вопроса с помощью LLM."""
    if not client_llm:
        print("Клиент LLM не инициализирован. Невозможно выбрать теги.")
        return tuple()
        
    tags_list = ", ".join(all_tags)
    prompt = f"""Ты — точный и строгий ИИ-классификатор. Твоя задача — из списка тегов выбрать все, которые относятся к вопросу пользователя.

СПИСОК ТЕГОВ: [{tags_list}]

ВОПРОС: {question}

Ответь ТОЛЬКО списком релевантных тегов, разделенных через "_|_" без пробелов. Например: тег1_|_тег2.
Если ни один тег не подходит, верни пустую строку."""

    completion = client_llm.chat.completions.create(
        model=LLM_MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0
    )
    content = completion.choices[0].message.content.strip()
    if not content:
        return tuple()
    raw_tags = [tag.strip() for tag in content.split('_|_') if tag.strip()]
    selected_tags = tuple(tag for tag in raw_tags if tag in all_tags)
    return selected_tags

async def generate_final_answer_async(question: str, knowledge_base: KnowledgeBase, all_tags: Tuple[str, ...]) -> str:
    """Генерирует итоговый ответ на вопрос, используя полный RAG-пайплайн."""
    if not client_llm:
        return "Ошибка: Клиент LLM не инициализирован."

    # 1. Маршрутизатор
    prompt = f"""Ты — ИИ-маршрутизатор. Определи, нужен ли для ответа на вопрос доступ к базе знаний банка.
- Ответь "да", если вопрос о банковских продуктах, услугах, тарифах, документах.
- Ответь "нет", если это общее приветствие, прощание или вопрос не по теме банка.

Вопрос: {question}

Ответь одним словом: "да" или "нет"."""
    completion_router = client_llm.chat.completions.create(
        model=LLM_MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0
    )
    router_result = completion_router.choices[0].message.content.strip().lower()

    # 2. Простой ответ
    if "нет" in router_result:
        prompt_simple = f"""Ты — дружелюбный ассистент банка. Кратко и вежливо ответь на реплику клиента.
Реплика: {question}
Ответ:"""
        completion_simple = client_llm.chat.completions.create(
            model=LLM_MODEL_NAME,
            messages=[{"role": "user", "content": prompt_simple}],
            temperature=0.3,
            max_tokens=150
        )
        return completion_simple.choices[0].message.content.strip()

    # 3. RAG
    tags = choose_tags_LLM(all_tags, question)
    relevant_documents = await find_relevant_docs_async(
        question=question,
        selected_tags=tags,
        docs_df=knowledge_base.docs_df,
        chunks_df=knowledge_base.chunks_df,
        top_k=3
    )
    if relevant_documents:
        context = "\n\n---\n\n".join(doc['text'] for doc in relevant_documents)
    else:
        context = "Нет релевантной информации."

    prompt_rag = f"""Ты — ассистент нашего банка.
**Твоя задача — отвечать на вопросы, используя ИСКЛЮЧИТЕЛЬНО предоставленную информацию (контекст). Запрещено использовать общие знания или придумывать детали.**
**Твои действия:**
1.  **Если ответ есть в контексте:** Четко и по делу сформулируй его на основе предоставленной информации.
2.  **Если ответа в контексте НЕТ:** Не пытайся помочь, придумывая ответ. Твоя единственная задача — вежливо направить клиента к официальным источникам. Используй шаблон:
    *   **Шаблон ответа:** "Извините, но у меня нет ответа на ваш вопрос. Чтобы получить актуальную и точную информацию о [тема вопроса], пожалуйста, обратиться в поддержку чата к специалисту"
**ЗАПРЕТЫ:**
*   Никогда не упоминай другие банки.
*   Никогда не говори клиенту про "контекст" или "базу знаний".

---

КОНТЕКСТ:
{context}

ВОПРОС: {question}

ОТВЕТ:"""
    completion_rag = client_llm.chat.completions.create(
        model=LLM_MODEL_NAME,
        messages=[{"role": "user", "content": prompt_rag}],
        temperature=0.1,
        max_tokens=500
    )
    return completion_rag.choices[0].message.content.strip()

# --- Логика из baseline/baseline.py ---

def baseline_answer_generation(question: str) -> str:
    """
    Функция для генерации ответа по заданному вопросу из baseline.py.
    """
    if not client_llm:
        return "Ошибка: Клиент LLM не инициализирован."
        
    response = client_llm.chat.completions.create(
        model=LLM_MODEL_NAME,
        messages=[
            {"role": "user", "content": f"Ответь на вопрос: {question}"}
        ]
    )
    return response.choices[0].message.content

def run_baseline():
    """
    Запускает логику из baseline/baseline.py.
    """
    print("--- ЗАПУСК РЕЖИМА BASELINE ---")
    try:
        questions_df = pd.read_csv(QUESTIONS_PATH)
        questions_list = questions_df['Вопрос'].tolist()
        answer_list = []
        for question in tqdm(questions_list, desc="Генерация ответов (baseline)"):
            answer = baseline_answer_generation(question=question)
            answer_list.append(answer)
        questions_df['Ответы на вопрос'] = answer_list
        questions_df.to_csv(SUBMISSION_PATH, index=False)
        print(f"Результаты baseline сохранены в '{SUBMISSION_PATH}'.")
    except FileNotFoundError:
        print(f"Ошибка: Файл с вопросами '{QUESTIONS_PATH}' не найден.")

# --- Основная логика и запуск (из main.py) ---

async def run_test_async():
    """
    Запускает небольшой асинхронный тест на нескольких вопросах и ограниченной базе знаний.
    """
    print("--- ЗАПУСК ТЕСТОВОГО РЕЖИМА ---")
    print("\n--- Шаг 1: Создание тестовой базы знаний (10 документов) ---")
    docs_df, chunks_df = await create_rag_database_async(row_limit=1)
    if docs_df.empty or chunks_df.empty:
        print("Не удалось создать тестовую базу знаний. Завершение теста.")
        return

    all_tags = get_all_tags(docs_df)
    knowledge_base = KnowledgeBase(docs_df=docs_df, chunks_df=chunks_df)
    test_questions = [
        "Как оформить кредитную карту?",
        "Какие есть вклады?",
        "Спасибо, до свидания"
    ]
    print(f"\n--- Шаг 2: Тестирование на {len(test_questions)} вопросах ---")
    for i, question in enumerate(test_questions):
        print(f"\n--- Вопрос {i+1}/{len(test_questions)} ---")
        print(f"Вопрос: {question}")
        answer = await generate_final_answer_async(question=question, knowledge_base=knowledge_base, all_tags=all_tags)
        print(f"Ответ: {answer}")
    print("\n--- ТЕСТОВЫЙ РЕЖИМ ЗАВЕРШЕН ---")

async def main_async():
    """
    Основная асинхронная функция для запуска всего RAG-пайплайна.
    """
    print("--- Шаг 1: Создание или загрузка базы знаний RAG ---")
    docs_df, chunks_df = await create_rag_database_async()
    if docs_df.empty or chunks_df.empty:
        print("Не удалось создать базу знаний. Завершение работы.")
        return

    all_tags = get_all_tags(docs_df)
    knowledge_base = KnowledgeBase(docs_df=docs_df, chunks_df=chunks_df)
    print(f"База знаний создана. Обнаружено {len(all_tags)} уникальных тегов.")

    print("\n--- Шаг 2: Чтение вопросов для обработки ---")
    try:
        questions_df = pd.read_csv(QUESTIONS_PATH)
        questions_list = questions_df['Вопрос'].tolist()
        print(f"Загружено {len(questions_list)} вопросов.")
    except FileNotFoundError:
        print(f"Ошибка: Файл с вопросами '{QUESTIONS_PATH}' не найден. Завершение работы.")
        return

    print("\n--- Шаг 3: Генерация ответов на вопросы ---")
    answer_list = []
    for question in tqdm(questions_list, desc="Генерация ответов"):
        answer = await generate_final_answer_async(
            question=question,
            knowledge_base=knowledge_base,
            all_tags=all_tags
        )
        answer_list.append(answer)

    print("\n--- Шаг 4: Сохранение результатов ---")
    questions_df['Ответы на вопрос'] = answer_list
    questions_df.to_csv(SUBMISSION_PATH, index=False)
    print(f"Результаты успешно сохранены в файл '{SUBMISSION_PATH}'.")


if __name__ == "__main__":
    if "--test" in sys.argv:
        asyncio.run(run_test_async())
    elif "--baseline" in sys.argv:
        run_baseline()
    else:
        asyncio.run(main_async())
