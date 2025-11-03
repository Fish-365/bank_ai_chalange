import os
from dotenv import load_dotenv
from openai import OpenAI
from read_base import find_relevant_docs
from typing import Tuple, NamedTuple
import pandas as pd

# --- Типы данных ---
class KnowledgeBase(NamedTuple):
    docs_df: pd.DataFrame
    chunks_df: pd.DataFrame

# --- Настройка клиента и моделей ---
load_dotenv()

LLM_API_KEY = os.getenv("LLM_API_KEY")
LLM_MODEL_NAME = "openrouter/mistralai/mistral-small-3.2-24b-instruct"

client_LLM = OpenAI(
    base_url="https://ai-for-finance-hack.up.railway.app/",
    api_key=LLM_API_KEY
)

print(f"Используемый API ключ для LLM: {LLM_API_KEY[-4:] if LLM_API_KEY else 'Не найден'}")


def choose_tags_LLM(all_tags: Tuple[str, ...], question: str) -> Tuple[str, ...]:
    """Выбирает релевантные теги для вопроса с помощью LLM."""
    tags_list = ", ".join(all_tags)

    prompt = f"""Ты — точный и строгий ИИ-классификатор. Твоя задача — из списка тегов выбрать все, которые относятся к вопросу пользователя.

СПИСОК ТЕГОВ: [{tags_list}]

ВОПРОС: {question}

Ответь ТОЛЬКО списком релевантных тегов, разделенных через "_|_" без пробелов. Например: тег1_|_тег2.
Если ни один тег не подходит, верни пустую строку."""

    completion = client_LLM.chat.completions.create(
        model=LLM_MODEL_NAME,
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.0
    )

    content = completion.choices[0].message.content.strip()
    if not content:
        return tuple()
        
    raw_tags = [tag.strip() for tag in content.split('_|_') if tag.strip()]
    # Фильтруем на случай, если модель вернет тег, которого нет в исходном списке
    selected_tags = tuple(tag for tag in raw_tags if tag in all_tags)
    return selected_tags


def generate_final_answer(question: str, knowledge_base: KnowledgeBase, all_tags: Tuple[str, ...]) -> str:
    """
    Генерирует итоговый ответ на вопрос, используя полный RAG-пайплайн.
    """
    # 1. Маршрутизатор: определяем, нужен ли RAG
    prompt_router = f"""Ты — ИИ-маршрутизатор. Определи, нужен ли для ответа на вопрос доступ к базе знаний банка.
- Ответь "да", если вопрос о банковских продуктах, услугах, тарифах, документах.
- Ответь "нет", если это общее приветствие, прощание или вопрос не по теме банка.

Вопрос: {question}

Ответь одним словом: "да" или "нет"."""

    completion_router = client_LLM.chat.completions.create(
        model=LLM_MODEL_NAME,
        messages=[{"role": "user", "content": prompt_router}],
        temperature=0.0
    )
    router_result = completion_router.choices[0].message.content.strip().lower()

    # 2. Если RAG не нужен, генерируем простой ответ
    if "нет" in router_result:
        prompt_simple = f"""Ты — дружелюбный ассистент банка. Кратко и вежливо ответь на реплику клиента.

Реплика: {question}

Ответ:"""
        completion_simple = client_LLM.chat.completions.create(
            model=LLM_MODEL_NAME,
            messages=[{"role": "user", "content": prompt_simple}],
            temperature=0.3,
            max_tokens=150
        )
        return completion_simple.choices[0].message.content.strip()

    # 3. Если RAG нужен, выполняем поиск и генерацию
    tags = choose_tags_LLM(all_tags, question)

    relevant_documents = find_relevant_docs(
        question=question,
        selected_tags=tags,
        docs_df=knowledge_base.docs_df,
        chunks_df=knowledge_base.chunks_df,
        top_k=3,
        vector_weight=0.7,
        tag_weight=0.3
    )

    if relevant_documents:
        context = "\n\n---\n\n".join(doc['text'] for doc in relevant_documents)
    else:
        context = "Нет релевантной информации."

    prompt_rag = f"""Ты — профессиональный ассистент банка. Твоя задача — дать точный и ясный ответ на вопрос клиента, основываясь ИСКЛЮЧИТЕЛЬНО на предоставленном контексте.

ИНСТРУКЦИИ:
1. Внимательно изучи контекст.
2. Сформулируй ответ на основе найденной информации.
3. Если контекст не содержит ответа на вопрос, скажи: "К сожалению, у меня нет информации по вашему вопросу. Рекомендую обратиться в чат поддержки банка для консультации со специалистом."
4. Не придумывай информацию и не используй свои общие знания.

КОНТЕКСТ:
{context}

ВОПРОС: {question}

ОТВЕТ:"""

    completion_rag = client_LLM.chat.completions.create(
        model=LLM_MODEL_NAME,
        messages=[{"role": "user", "content": prompt_rag}],
        temperature=0.1,
        max_tokens=500
    )
    return completion_rag.choices[0].message.content.strip()