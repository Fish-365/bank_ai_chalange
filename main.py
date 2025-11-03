import pandas as pd
from tqdm import tqdm
import sys
from typing import NamedTuple

# Импортируем наши кастомные функции
from read_base import create_rag_database, get_all_tags
from model import generate_final_answer

# --- Глобальные переменные и константы ---
TRAIN_DATA_PATH = 'baseline/train_data.csv'
QUESTIONS_PATH = './questions.csv'
SUBMISSION_PATH = 'submission.csv'

# --- Типы данных для улучшения читаемости ---
class KnowledgeBase(NamedTuple):
    docs_df: pd.DataFrame
    chunks_df: pd.DataFrame

def run_test():
    """
    Запускает небольшой тест на нескольких вопросах и ограниченной базе знаний.
    """
    print("--- ЗАПУСК ТЕСТОВОГО РЕЖИМА ---")
    
    print("\n--- Шаг 1: Создание тестовой базы знаний (10 документов) ---")
    docs_df, chunks_df = create_rag_database(file_path=TRAIN_DATA_PATH, row_limit=10)

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
        answer = generate_final_answer(question=question, knowledge_base=knowledge_base, all_tags=all_tags)
        print(f"Ответ: {answer}")
    
    print("\n--- ТЕСТОВЫЙ РЕЖИМ ЗАВЕРШЕН ---")

def main():
    """
    Основная функция для запуска всего RAG-пайплайна.
    """
    print("--- Шаг 1: Создание или загрузка базы знаний RAG ---")
    docs_df, chunks_df = create_rag_database(file_path=TRAIN_DATA_PATH)

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
        answer = generate_final_answer(
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
        run_test()
    else:
        main()
