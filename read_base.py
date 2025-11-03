import pandas as pd
import re
from tqdm import tqdm
import numpy as np
from embedding_model import get_embedding
from typing import Optional, Tuple, List, Dict, Any, Set

# --- Параметры для API запросов ---
MAX_API_RETRIES = 5
INITIAL_API_BACKOFF = 1

# === ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ===

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
    """
    Собирает все уникальные теги из DataFrame документов.

    :param docs_df: DataFrame с колонкой 'tags'.
    :return: Кортеж уникальных тегов.
    """
    if 'tags' not in docs_df.columns:
        return tuple()
    
    all_tags: Set[str] = set()
    for tags_str in docs_df['tags'].dropna():
        all_tags.update(tag.strip() for tag in tags_str.split(','))
    return tuple(sorted(list(all_tags)))


# === ОСНОВНОЙ КОД ===

def create_rag_database(file_path: str = 'baseline/train_data.csv', row_limit: Optional[int] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Создает базу данных для RAG из CSV-файла.

    :param file_path: Путь к исходному CSV-файлу.
    :param row_limit: Опциональное ограничение на количество строк для обработки.
    :return: Кортеж из двух DataFrame: (docs_df, chunks_df)
    """
    pattern = r'Обновлено \d{2}\.\d{2}\.\d{4} в \d{1,2}:\d{2}'
    question_pattern = r'^##.*$'
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

    final_result = []
    for item in tqdm(items_to_process, desc="Получение эмбеддингов"):
        embedding = get_embedding(
            item['cleaned'],
            max_retries=MAX_API_RETRIES,
            initial_backoff=INITIAL_API_BACKOFF
        )

        if embedding is None:
            continue
        
        final_result.append((item['doc_id'], item['original'], item['cleaned'], embedding))
            
    if not final_result:
        print("\nЭмбеддинги не были созданы. Возвращается пустая база данных.")
        return empty_dfs

    docs_df = pd.DataFrame(documents_data)
    chunks_df = pd.DataFrame(final_result, columns=['doc_id', 'original_chunk', 'cleaned_chunk', 'embedding'])
    print(f"\nОбработка завершена. Создано {len(docs_df)} документов и {len(chunks_df)} чанков с эмбеддингами.")
    
    return docs_df, chunks_df


def find_relevant_docs(question: str, selected_tags: tuple, docs_df: pd.DataFrame, chunks_df: pd.DataFrame, top_k: int = 5, vector_weight: float = 0.7, tag_weight: float = 0.3) -> List[Dict[str, Any]]:
    """
    Находит наиболее релевантные документы, используя гибридный поиск.
    """
    if chunks_df.empty:
        return []

    question_embedding = get_embedding(question)
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
