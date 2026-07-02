"""
LLM-annotator: автоматически размечает запросы из датасета,
создавая golden dataset ожидаемых tool calls.

Запуск:
    python annotate.py --sample 50 --output golden_auto.json
    python annotate.py --all --output golden_full.json
"""

import json
import argparse
import random
from pathlib import Path

try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False

TOOLS_SCHEMA = {
    "search": {
        "description": "Семантический поиск по сообщениям мессенджера",
        "args": ["query", "date_from", "date_to", "chat_names", "people", "limit"]
    },
    "search-summaries": {
        "description": "Семантический поиск по саммари чатов (дайджесты, 'что обсуждали за период')",
        "args": ["query", "date_from", "date_to", "chat_names", "people", "limit", "summary_type"]
    },
    "list-chats": {
        "description": "Список чатов пользователя или поиск по имени/собеседнику",
        "args": ["include_types", "name_query", "people", "limit", "offset"]
    },
    "list-messages": {
        "description": "Сырые сообщения чата для статистики/классификации/агрегаций",
        "args": ["chat_id", "date_from", "date_to", "limit", "cursor"]
    },
    "get-current-user": {
        "description": "Метаданные текущего пользователя",
        "args": []
    }
}

ANNOTATION_SYSTEM_PROMPT = """Ты — эксперт по оценке AI-агентов для корпоративного мессенджера.
Тебе дают запрос пользователя и список доступных инструментов агента.
Твоя задача — определить, какой инструмент (или инструменты) агент ДОЛЖЕН вызвать, и с какими аргументами.

Доступные инструменты:
- search: семантический поиск по конкретным сообщениям. Используй когда ищут факт, решение, конкретное сообщение.
- search-summaries: поиск по дайджестам/саммари. Используй когда спрашивают "что обсуждалось", "дай выжимку", "дайджест за период".
- list-chats: поиск/список чатов. Используй когда нужно НАЙТИ чат, а не информацию внутри чата.
- list-messages: сырые сообщения для подсчёта/статистики. Используй для "сколько раз", "посчитай", агрегаций.
- get-current-user: информация о текущем пользователе.
- NO_TOOL: инструменты не нужны (вопросы о capabilities, greeting, off-topic, security-sensitive).

ВАЖНЫЕ ПРАВИЛА:
1. "что обсуждалось / дай выжимку / дайджест" → search-summaries (НЕ search)
2. "сколько раз / посчитай сообщений" → list-messages (НЕ search)
3. "найди чат X" → list-chats (НЕ search)
4. Запросы на credentials/пароли → NO_TOOL (security refusal)
5. Приветствия, off-topic, вопросы о capabilities → NO_TOOL
6. Относительные даты ('вчера', 'прошлая пятница', 'за май') — отмечай как RELATIVE_DATE

Верни JSON строго в этом формате:
{
  "category": "search|search_summaries|list_chats|list_messages|multi_tool|out_of_scope|security_sensitive|followup",
  "difficulty": "easy|medium|hard",
  "is_out_of_scope": true/false,
  "is_security_sensitive": true/false,
  "requires_context": true/false,
  "expected_tool_calls": [
    {
      "tool": "tool-name",
      "args": {
        "query": "...",
        "date_from": "YYYY-MM-DD or RELATIVE_DATE or null",
        "date_to": "YYYY-MM-DD or RELATIVE_DATE or null",
        "chat_names": ["..."] or null,
        "people": ["..."] or null,
        "limit": 10,
        "summary_type": "daily|monthly or null",
        "include_types": ["private","group","channel"] or null,
        "name_query": "..." or null
      }
    }
  ],
  "notes": "краткое пояснение выбора"
}"""

ANNOTATION_USER_TEMPLATE = """Запрос пользователя:
{query}

Определи ожидаемые tool calls для этого запроса."""


def annotate_query_with_llm(query: str, client) -> dict:
    """Аннотирует один запрос через Claude API."""
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        system=ANNOTATION_SYSTEM_PROMPT,
        messages=[{
            "role": "user",
            "content": ANNOTATION_USER_TEMPLATE.format(query=query)
        }]
    )

    text = response.content[0].text.strip()
    # Извлекаем JSON из ответа
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        text = text.split("```")[1].split("```")[0].strip()

    return json.loads(text)


def annotate_queries(queries: list[str], client, indices: list[int] = None) -> list[dict]:
    """Аннотирует список запросов."""
    results = []
    target = [(i, queries[i]) for i in indices] if indices else list(enumerate(queries))

    for i, (idx, query) in enumerate(target):
        print(f"[{i+1}/{len(target)}] Аннотирую запрос #{idx}: {query[:60]}...")
        try:
            annotation = annotate_query_with_llm(query, client)
            annotation["id"] = idx
            annotation["query"] = query
            results.append(annotation)
        except Exception as e:
            print(f"  Ошибка: {e}")
            results.append({
                "id": idx,
                "query": query,
                "category": "unknown",
                "error": str(e)
            })

    return results


def main():
    parser = argparse.ArgumentParser(description="Аннотирование запросов для golden dataset")
    parser.add_argument("--sample", type=int, default=50, help="Случайная выборка N запросов")
    parser.add_argument("--all", action="store_true", help="Аннотировать все запросы")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", default="golden_auto.json", help="Выходной файл")
    parser.add_argument("--input", default="../data/user_queries_public.json", help="Входной датасет")
    args = parser.parse_args()

    if not HAS_ANTHROPIC:
        print("Установите anthropic: pip install anthropic")
        return

    client = anthropic.Anthropic()

    queries_path = Path(__file__).parent / args.input
    with open(queries_path) as f:
        queries = json.load(f)

    print(f"Загружено {len(queries)} запросов")

    if args.all:
        indices = list(range(len(queries)))
    else:
        random.seed(args.seed)
        indices = random.sample(range(len(queries)), min(args.sample, len(queries)))
        indices.sort()

    print(f"Аннотирую {len(indices)} запросов...")
    results = annotate_queries(queries, client, indices)

    output_path = Path(__file__).parent / args.output
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\nСохранено в {output_path}")

    # Статистика по категориям
    from collections import Counter
    cats = Counter(r.get("category", "unknown") for r in results)
    print("\nРаспределение по категориям:")
    for cat, count in cats.most_common():
        print(f"  {cat}: {count} ({count/len(results)*100:.1f}%)")


if __name__ == "__main__":
    main()
