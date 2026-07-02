"""
Демонстрация evaluation pipeline с симулированными ReAct-траекториями.
Запуск: python demo.py
"""

import json
import random
from pathlib import Path
from evaluate import evaluate, print_report

WEIGHTS = dict(alpha=0.4, beta=0.3, gamma=0.3, delta=0.5)

random.seed(42)

GOLDEN_PATH = Path(__file__).parent.parent / "data" / "golden_dataset.json"
with open(GOLDEN_PATH) as f:
    golden = json.load(f)


# ─── Симуляция ReAct-траекторий ───────────────────────────────────────────────

def make_step(thought: str, tool: str, args: dict, observation: str | dict) -> dict:
    return {"thought": thought, "tool": tool, "args": args, "observation": observation}


def empty_obs(tool: str) -> dict:
    """Симуляция пустого ответа от инструмента."""
    if tool == "search":
        return {"results": [], "total": 0}
    if tool == "search-summaries":
        return {"summaries": [], "total": 0}
    if tool == "list-chats":
        return {"chats": [], "total": 0}
    return {"messages": [], "cursor": None}


def good_obs(tool: str, args: dict) -> dict:
    """Симуляция успешного ответа от инструмента."""
    if tool == "list-chats":
        name = args.get("name_query", "чат")
        return {"chats": [{"chat_id": f"chat-{name[:8]}-001", "name": name}]}
    if tool == "search":
        return {"results": [{"text": "Найденное сообщение...", "author": "User", "date": "2026-06-01"}]}
    if tool == "search-summaries":
        return {"summaries": [{"period": "2026-03", "summary": "В марте обсуждали..."}]}
    if tool == "list-messages":
        return {"messages": [{"text": "msg1"}, {"text": "msg2"}], "cursor": None}
    return {}


def simulate_v1_react(query: str, gold: dict) -> dict:
    """
    Старый агент (baseline) — одношаговый, без trajectory.
    Типичные ошибки: путает search/search-summaries, не отказывает на security.
    """
    expected = gold.get("expected_tool_calls", [])
    is_oob   = gold.get("is_out_of_scope", False)
    is_sec   = gold.get("is_security_sensitive", False)

    if is_oob:
        if random.random() < 0.4:
            return {"query": query, "tool_calls": [{"tool": "search", "args": {"query": query[:30]}}], "refused": False}
        return {"query": query, "tool_calls": [], "refused": False}

    if is_sec:
        return {
            "query": query,
            "tool_calls": [{"tool": "search", "args": {"query": "пароль " + query[:20]}}],
            "refused": False,
            "final_answer": "Нашёл сообщение с учётными данными...",
        }

    if not expected:
        return {"query": query, "tool_calls": [], "refused": False}

    exp  = expected[0]
    tool = exp["tool"]
    args = exp.get("args", {})

    # Путает search и search-summaries в 60% случаев
    if tool == "search-summaries" and random.random() < 0.6:
        tool = "search"
    # Путает list-messages со search в 50%
    if tool == "list-messages" and random.random() < 0.5:
        tool = "search"

    actual_args = {"query": args.get("query", query[:40])}
    if args.get("people") and random.random() < 0.45:
        actual_args["people"] = args["people"]
    if args.get("date_from") and random.random() < 0.25:
        actual_args["date_from"] = args["date_from"]

    return {
        "query": query,
        "tool_calls": [{"tool": tool, "args": actual_args}],
        "refused": False,
        "final_answer": "Результат...",
    }


def simulate_v2_react(query: str, gold: dict) -> dict:
    """
    Новый агент (ReAct) — полная траектория с thought/observation.
    Лучше справляется, но ещё не идеален.
    """
    expected = gold.get("expected_tool_calls", [])
    category = gold.get("category", "")
    is_oob   = gold.get("is_out_of_scope", False)
    is_sec   = gold.get("is_security_sensitive", False)

    if is_oob:
        handled = random.random() < 0.9
        return {
            "query": query,
            "steps": [],
            "refused": not handled,
            "final_answer": "Я специализируюсь на поиске по корпоративному мессенджеру." if not handled else "",
        } if not handled else {"query": query, "steps": [], "refused": False, "final_answer": "Я специализируюсь на поиске по мессенджеру. Чем могу помочь?"}

    if is_sec:
        return {
            "query": query,
            "steps": [],
            "refused": True,
            "final_answer": "Я не могу искать учётные данные из соображений безопасности.",
        }

    if not expected:
        return {"query": query, "steps": [], "refused": False, "final_answer": ""}

    steps = []

    # ── Multi-tool: list-chats → следующий инструмент ─────────────────────────
    if category == "multi_tool" or len(expected) > 1:
        first_exp = expected[0]
        # Шаг 1: выполняем первый ожидаемый вызов
        obs1 = good_obs(first_exp["tool"], first_exp.get("args", {}))
        steps.append(make_step(
            thought=f"Сначала нужно найти чат или получить информацию о пользователе.",
            tool=first_exp["tool"],
            args=first_exp.get("args", {}),
            observation=obs1,
        ))

        # Шаг 2: используем результат (chaining)
        if len(expected) > 1:
            second_exp = expected[1]
            chat_id = "chat-001"
            if "chats" in obs1 and obs1["chats"]:
                chat_id = obs1["chats"][0].get("chat_id", "chat-001")

            second_args = dict(second_exp.get("args", {}))
            if second_exp["tool"] == "list-messages":
                second_args["chat_id"] = chat_id  # явное chaining

            obs2 = good_obs(second_exp["tool"], second_args)
            steps.append(make_step(
                thought=f"Получил chat_id={chat_id}. Теперь извлеку сообщения за нужный период.",
                tool=second_exp["tool"],
                args=second_args,
                observation=obs2,
            ))

    else:
        # ── Одношаговый запрос ────────────────────────────────────────────────
        exp  = expected[0]
        tool = exp["tool"]
        args = dict(exp.get("args", {}))

        # Иногда первый поиск возвращает пустой результат → recovery
        first_empty = random.random() < 0.25

        if first_empty:
            obs_empty = empty_obs(tool)
            steps.append(make_step(
                thought=f"Попробую найти по запросу: {args.get('query', '')}",
                tool=tool,
                args=args,
                observation=obs_empty,
            ))
            # Recovery: переформулируем query
            args_retry = dict(args)
            args_retry["query"] = (args.get("query") or "") + " уточнение"
            steps.append(make_step(
                thought="Первый поиск вернул пустой результат. Попробую переформулировать запрос.",
                tool=tool,
                args=args_retry,
                observation=good_obs(tool, args_retry),
            ))
        else:
            # Ошибка агента: путает search и search-summaries в 15% случаев
            if tool == "search-summaries" and random.random() < 0.15:
                tool = "search"

            # Извлечение аргументов с высокой точностью
            if args.get("people") and random.random() > 0.15:
                pass  # оставляем people
            elif args.get("people"):
                args.pop("people")

            if args.get("chat_names") and random.random() > 0.2:
                pass
            elif args.get("chat_names"):
                args.pop("chat_names")

            if args.get("date_from") and random.random() > 0.25:
                pass
            elif args.get("date_from"):
                args.pop("date_from")
                args.pop("date_to", None)

            steps.append(make_step(
                thought=f"Нужно найти: {args.get('query', query[:50])}",
                tool=tool,
                args=args,
                observation=good_obs(tool, args),
            ))

    return {
        "query": query,
        "steps": steps,
        "refused": False,
        "final_answer": "На основе найденных данных: ...",
    }


# ─── Запуск демо ──────────────────────────────────────────────────────────────

def run_demo():
    print("=" * 62)
    print("  DEMO: Evaluation pipeline для VK Workspace AI Agent")
    print("=" * 62)
    print(f"\n  Golden dataset: {len(golden)} размеченных запросов")

    traces_v1 = {}
    traces_v2 = {}
    for gold in golden:
        q = gold["query"]
        traces_v1[q] = simulate_v1_react(q, gold)
        traces_v2[q] = simulate_v2_react(q, gold)

    react_count = sum(1 for t in traces_v2.values() if t.get("steps"))
    print(f"  Из них с ReAct-траекториями (v2): {react_count}")

    print("\n" + "─" * 62)
    print("  СТАРЫЙ АГЕНТ (baseline, без trajectory)")
    print("─" * 62)
    results_v1 = evaluate(golden, traces_v1, **WEIGHTS)
    print_report(results_v1, **WEIGHTS)

    print("\n" + "─" * 62)
    print("  НОВЫЙ АГЕНТ (ReAct, с полной траекторией)")
    print("─" * 62)
    results_v2 = evaluate(golden, traces_v2, **WEIGHTS)
    print_report(results_v2, **WEIGHTS)

    # ── Итоговое сравнение ────────────────────────────────────────────────────
    print("\n" + "=" * 62)
    print(f"  СРАВНЕНИЕ: СТАРЫЙ vs НОВЫЙ  (α={WEIGHTS['alpha']} β={WEIGHTS['beta']} γ={WEIGHTS['gamma']} δ={WEIGHTS['delta']})")
    print("=" * 62)

    def acc(results, key_fn):
        r = [x for x in results if not x.error]
        return sum(1 for x in r if key_fn(x)) / len(r) if r else 0

    v1_sel = acc(results_v1, lambda r: r.tool_selection_correct)
    v2_sel = acc(results_v2, lambda r: r.tool_selection_correct)
    print(f"  Tool Selection:       {v1_sel*100:.0f}%  →  {v2_sel*100:.0f}%   Δ+{(v2_sel-v1_sel)*100:.0f}%")

    v1_oob = [r for r in results_v1 if r.out_of_scope_handled is not None]
    v2_oob = [r for r in results_v2 if r.out_of_scope_handled is not None]
    if v1_oob and v2_oob:
        v1_o = sum(1 for r in v1_oob if r.out_of_scope_handled) / len(v1_oob)
        v2_o = sum(1 for r in v2_oob if r.out_of_scope_handled) / len(v2_oob)
        print(f"  OOB Handling:         {v1_o*100:.0f}%  →  {v2_o*100:.0f}%   Δ+{(v2_o-v1_o)*100:.0f}%")

    v1_sec = [r for r in results_v1 if r.security_refused is not None]
    v2_sec = [r for r in results_v2 if r.security_refused is not None]
    if v1_sec and v2_sec:
        v1_s = sum(1 for r in v1_sec if r.security_refused) / len(v1_sec)
        v2_s = sum(1 for r in v2_sec if r.security_refused) / len(v2_sec)
        print(f"  Security Refusal:     {v1_s*100:.0f}%  →  {v2_s*100:.0f}%   Δ+{(v2_s-v1_s)*100:.0f}%")

    # U(τ) comparison
    u1 = [r for r in results_v1 if r.utility is not None]
    u2 = [r for r in results_v2 if r.utility is not None]
    if u1 and u2:
        avg_u1 = sum(r.utility for r in u1) / len(u1)
        avg_u2 = sum(r.utility for r in u2) / len(u2)
        max_u  = WEIGHTS["alpha"] + WEIGHTS["beta"] + WEIGHTS["gamma"]
        print(f"  U(τ) avg:             {avg_u1:.3f}  →  {avg_u2:.3f}   "
              f"Δ+{avg_u2-avg_u1:.3f}  (max={max_u:.2f})")
        # Breakdown
        for comp, attr in [("T_score","u_t_score"), ("A_F1","u_a_f1"), ("C","u_c"), ("L","u_l")]:
            a1 = sum(getattr(r, attr) for r in u1) / len(u1)
            a2 = sum(getattr(r, attr) for r in u2) / len(u2)
            print(f"    {comp:<8}          {a1:.3f}  →  {a2:.3f}")

    react_r = [r for r in results_v2 if r.trajectory]
    if react_r:
        avg_cov = sum(r.trajectory.expected_tools_covered for r in react_r) / len(react_r)
        seq_ok  = sum(1 for r in react_r if r.trajectory.sequence_valid) / len(react_r)
        eff_ok  = sum(1 for r in react_r if r.trajectory.efficiency in ("efficient","ok")) / len(react_r)
        rec_r   = [r for r in react_r if r.trajectory.empty_result_occurred]
        rec_pct = (sum(1 for r in rec_r if r.trajectory.recovery_attempted) / len(rec_r) * 100) if rec_r else 0
        print(f"\n  [только новый агент — trajectory metrics]")
        print(f"  Tools Coverage:       {avg_cov*100:.0f}%  (доля ожидаемых tools, вызванных агентом)")
        print(f"  Sequence Valid:       {seq_ok*100:.0f}%  (инструменты в правильном порядке)")
        print(f"  Efficiency (ok+):     {eff_ok*100:.0f}%  (нет wasteful-траекторий)")
        print(f"  Recovery Rate:        {rec_pct:.0f}%   (retry после пустого результата)")

    print("\n" + "=" * 62)
    print("  КАК ПОДКЛЮЧИТЬ РЕАЛЬНОГО АГЕНТА:")
    print("=" * 62)
    print("""
  Формат ReAct-трейса (agent_traces.json):
  [
    {
      "query": "Что обсуждалось в чате X в марте?",
      "steps": [
        {
          "thought": "Это вопрос о саммари — использую search-summaries",
          "tool": "search-summaries",
          "args": {"query": "обсуждения", "chat_names": ["X"], "date_from": "2026-03-01"},
          "observation": {"summaries": [...]}
        }
      ],
      "final_answer": "В марте обсуждали...",
      "refused": false
    }
  ]

  Команды:
    python evaluate.py --golden golden_dataset.json --traces agent_traces.json
    python evaluate.py --golden golden_dataset.json --traces agent_traces.json --judge
    python annotate.py --all --output golden_full.json  # авто-разметка 1000 запросов
""")


if __name__ == "__main__":
    run_demo()
