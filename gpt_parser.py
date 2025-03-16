import tiktoken
import requests
import json
from openai import OpenAI
import time
from typing import Dict, Any
from dotenv import dotenv_values


# Загрузка переменных окружения
config = dotenv_values(".env")

# API ключ
OPENAI_API_KEY = config["OPENAI_API_KEY"]

def scrape_html(url: str) -> str:
    """Получение HTML-кода страницы"""
    response = requests.get(url)
    return response.text

def extract_info(content: str, client: OpenAI, model: str = "gpt-4o"):
    """Извлечение информации с помощью GPT"""
    # Прописываем системный промпт
    system_message = {
        "role": "system",
        "content": "Получи цену и рейтинг на все книги со страницы строго в json формате: {book: str, price: float, rating: int}."
    }
    messages = [system_message]
    # Добавляем в промпт HTML-код документа
    messages.append({"role": "user", "content": content})
    # Делаем запрос к API
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        # Формат ответа json
        response_format={"type": "json_object"}
    )
    # Возвращаем ответ
    return response.choices[0].message.content

def count_tokens(text: str, model: str = "gpt-4o") -> int:
    """Подсчет количества токенов в тексте"""
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

def calculate_cost(input_tokens: int, output_tokens: int, model: str = "gpt-4o") -> Dict[str, float]:
    """Расчет стоимости запроса на основе количества токенов"""
    rates = {
        "gpt-4o": {"input": 5, "output": 15},
        "gpt-3.5-turbo": {"input": 0.5, "output": 1.5}
    }
    
    input_cost = input_tokens * rates[model]["input"] / 1_000_000
    output_cost = output_tokens * rates[model]["output"] / 1_000_000
    
    return {
        "input_cost": input_cost,
        "output_cost": output_cost,
        "total_cost": input_cost + output_cost
    }

if __name__ == "__main__":
    URL = "http://books.toscrape.com/"

    MODEL = "gpt-4o"

    client = OpenAI(api_key=OPENAI_API_KEY)

    html_content = scrape_html(URL)
    input_tokens = count_tokens(html_content, MODEL)

    # Извлечение информации
    result = extract_info(html_content, client, MODEL)

    # Подсчет выходных токенов
    output_tokens = count_tokens(result, MODEL)

    # Расчет стоимости
    cost = calculate_cost(input_tokens, output_tokens, MODEL)

    # Отчет о стоимости
    print("\n--- ОТЧЕТ О СТОИМОСТИ ПАРСИНГА ---")
    print(f"Модель: {MODEL}")
    print(f"Входные токены: {input_tokens:,} (${cost['input_cost']:.4f})")
    print(f"Выходные токены: {output_tokens:,} (${cost['output_cost']:.4f})")
    print(f"ИТОГО: ${cost['total_cost']:.4f}")

    # Парсим JSON-ответ
    parsed_data = json.loads(result)

    # Вывод результатов
    print("\n--- РЕЗУЛЬТАТЫ ПАРСИНГА ---")
    print(f"Всего книг извлечено: {len(parsed_data['books'])}")
    print("\nПример данных (первые 3 книги):")
    for i, book in enumerate(parsed_data['books'][:3]):
        print(f"{i+1}. {book['book']} - £{book['price']} - {book['rating']} звезд")
        
    # Сохранение результатов в файл
    with open("parsed_books.json", "w", encoding="utf-8") as f:
        json.dump(parsed_data, f, ensure_ascii=False, indent=2)

    print(f"\nРезультаты сохранены в файл parsed_books.json")