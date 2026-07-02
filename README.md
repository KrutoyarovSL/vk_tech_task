# Evaluation Framework

Фреймворк для поведенческой оценки ReAct-агента без ground truth ответов.

## Проблема

Стандартные метрики (accuracy, BLEU) не применимы к ReAct-агентам: один и тот же вопрос может требовать разной цепочки действий в зависимости от контекста. Вместо оценки текста — оцениваем поведение: какой инструмент выбран, с какими аргументами, насколько эффективна траектория.

## Структура

```
├── data/
│   ├── user_queries_public.json   # 1000 реальных запросов пользователей
│   └── golden_dataset.json        # 30 вручную размеченных эталонов
├── eval/
│   ├── evaluate.py                # ядро фреймворка
│   ├── annotate.py                # LLM авто-разметка запросов
│   ├── demo.py                    # сравнение старого и нового агента (симуляция)
│   └── walkthrough.py             # пошаговый разбор одного запроса
└── assets/
    ├── pipeline_diagram.png       # схема пайплайна
    └── diagram.py                 # скрипт генерации схемы
```

## Метрики

### 1. Tool Selection Accuracy
Бинарная проверка: совпадает ли вызванный инструмент с эталоном.  
Критично, потому что `search` и `search-summaries` — принципиально разные запросы.

### 2. Argument Extraction F1 (A\_F1)
Macro-average F1 по аргументам, которые ожидались в golden:
- `people`, `chat_names` — F1 по спискам (precision + recall)
- `date_from`, `date_to` — бинарно
- `summary_type` — бинарно

### 3. Trajectory Evaluation
Специфика ReAct-агента — оцениваем всю цепочку шагов:
- **Coverage** — доля ожидаемых инструментов, вызванных агентом
- **Sequence validity** — порядок вызовов соответствует эталону
- **Recovery** — агент переформулировал запрос после пустого ответа (хорошо)
- **Loops (L)** — агент повторил идентичный вызов (плохо)
- **Chaining** — агент передал `chat_id` из одного шага в следующий

### 4. U(τ) — Hierarchical Trajectory Utility Function

$$U(\tau) = \alpha \cdot T_{score} + \beta \cdot A_{F1} + \gamma \cdot C - \delta \cdot L$$

| Компонент | Описание | Дефолт |
|-----------|----------|--------|
| $T_{score} \in \{0,1\}$ | Верный выбор инструмента | α = 0.4 |
| $A_{F1} \in [0,1]$ | F1 по аргументам | β = 0.3 |
| $C \in [0,1]$ | Path Convergence: $\min(1,\ S_{opt}/(S_{act}-R))$ | γ = 0.3 |
| $L \in \mathbb{N}$ | Счётчик зациклений | δ = 0.5 |

**Ключевое различие:** Recovery $R$ (умный retry) не штрафуется и вычитается из $S_{act}$, тогда как Loops $L$ штрафуются явно. Сумма $\alpha + \beta + \gamma = 1$, поэтому максимум $U(\tau) = 1.0$.

## Запуск

### Авто-разметка датасета (требует ANTHROPIC\_API\_KEY)

```bash
# Разметить 50 случайных запросов
python eval/annotate.py --sample 50 --output data/golden_auto.json

# Разметить все 1000
python eval/annotate.py --all --output data/golden_full.json
```

### Оценка агента

```bash
# Базовый запуск
python eval/evaluate.py \
    --golden data/golden_dataset.json \
    --traces agent_traces.json

# С кастомными весами и LLM-judge
python eval/evaluate.py \
    --golden data/golden_dataset.json \
    --traces agent_traces.json \
    --alpha 0.4 --beta 0.3 --gamma 0.3 --delta 0.5 \
    --judge \
    --output results.json
```

### Демо с симуляцией

```bash
# Сравнение старого (baseline) и нового (ReAct) агента
python eval/demo.py

# Пошаговый разбор одного запроса с формулами
python eval/walkthrough.py
```

## Формат трейсов агента

```json
[
  {
    "query": "Что обсуждалось в чате маркетинга за март?",
    "steps": [
      {
        "thought": "Вопрос о саммари — использую search-summaries",
        "tool": "search-summaries",
        "args": {
          "query": "обсуждения",
          "chat_names": ["маркетинг"],
          "date_from": "2026-03-01",
          "date_to": "2026-03-31",
          "summary_type": "monthly"
        },
        "observation": {"summaries": [...], "total": 1}
      }
    ],
    "final_answer": "В марте обсуждали...",
    "refused": false
  }
]
```

Поддерживается также упрощённый формат без `steps` (только `tool_calls`).

## Доступные инструменты агента

| Инструмент | Назначение |
|---|---|
| `search` | Семантический поиск по конкретным сообщениям |
| `search-summaries` | Поиск по дайджестам чатов («что обсуждалось за период») |
| `list-chats` | Список/поиск чатов пользователя |
| `list-messages` | Сырые сообщения для статистики и агрегаций |
| `get-current-user` | Метаданные текущего пользователя |

## Известные ограничения

- **Качество эталонов** зависит от LLM-аннотатора: для критичных случаев рекомендуется ручная проверка
- **Неоднозначные запросы** — некоторые запросы допускают несколько правильных инструментов; в таких случаях стоит расширить `expected_tool_calls` до списка допустимых вариантов
- **Относительные даты** («вчера», «на прошлой неделе») не проверяются без timestamp запроса
- **LLM-judge** при запуске на 1000 запросов требует значительных API-расходов; рекомендуется использовать выборочно
