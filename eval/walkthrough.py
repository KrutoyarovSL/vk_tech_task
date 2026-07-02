"""
Пошаговый разбор одного запроса через evaluation pipeline.
Запуск: python walkthrough.py
"""

import json
import sys
from evaluate import evaluate, compute_argument_f1, compute_utility

# ── Пример запроса ────────────────────────────────────────────────────────────

QUERY = "Что обсуждалось в чате маркетинга за последний месяц?"

GOLDEN = {
    "id": 1,
    "query": QUERY,
    "category": "search_summaries",
    "difficulty": "medium",
    "expected_tool_calls": [
        {
            "tool": "search-summaries",
            "args": {
                "query": "обсуждения маркетинг",
                "chat_names": ["маркетинг"],
                "date_from": "2026-06-01",
                "date_to": "2026-06-30",
                "summary_type": "monthly",
            },
        }
    ],
    "is_out_of_scope": False,
    "is_security_sensitive": False,
    "requires_context": False,
    "notes": "Ключевое слово 'что обсуждалось' → search-summaries, не search",
}

OLD_TRACE = {
    "query": QUERY,
    "tool_calls": [
        {
            "tool": "search",          # неправильный инструмент
            "args": {
                "query": "обсуждалось маркетинг",
                # нет: chat_names, date_from, date_to, summary_type
            },
        }
    ],
    "refused": False,
    "final_answer": "Нашёл несколько сообщений из чата маркетинга...",
}

NEW_TRACE = {
    "query": QUERY,
    "steps": [
        {
            "thought": (
                "Пользователь спрашивает 'что обсуждалось' — "
                "это запрос на дайджест/саммари. Нужен search-summaries, а не search. "
                "Вижу 'чат маркетинга' и 'последний месяц' — добавлю фильтры."
            ),
            "tool": "search-summaries",
            "args": {
                "query": "обсуждения маркетинг",
                "chat_names": ["маркетинг"],
                "date_from": "2026-06-01",
                "date_to": "2026-06-30",
                "summary_type": "monthly",
            },
            "observation": {
                "summaries": [
                    {
                        "period": "2026-06",
                        "summary": (
                            "В июне обсуждали запуск нового продукта, "
                            "бюджет на Q3 и редизайн лендинга."
                        ),
                    }
                ],
                "total": 1,
            },
        }
    ],
    "refused": False,
    "final_answer": (
        "В чате маркетинга за июнь обсуждали: "
        "запуск нового продукта, бюджет на Q3 и редизайн лендинга."
    ),
}

WEIGHTS = dict(alpha=0.4, beta=0.3, gamma=0.3, delta=0.5)

# ── Форматирование ────────────────────────────────────────────────────────────

W = 64

def hr(ch="─"):   print(ch * W)
def hr2(ch="═"):  print(ch * W)
def sec(title):   hr2(); print(f"  {title}"); hr2()
def sub(title):   hr(); print(f"  {title}"); hr()
def ok(s):        return f"\033[32m✓ {s}\033[0m"
def err(s):       return f"\033[31m✗ {s}\033[0m"
def warn(s):      return f"\033[33m~ {s}\033[0m"
def bold(s):      return f"\033[1m{s}\033[0m"


def print_args_diff(actual_args: dict, expected_args: dict):
    all_keys = sorted(set(list(actual_args) + list(expected_args)))
    for k in all_keys:
        exp = expected_args.get(k)
        act = actual_args.get(k)
        if exp is None:
            continue  # не ожидается — не проверяем
        if act == exp:
            print(f"    {ok(k):<30} {json.dumps(act, ensure_ascii=False)}")
        elif act is None:
            print(f"    {err(k + ' (отсутствует)'):<38} ожидалось: {json.dumps(exp, ensure_ascii=False)}")
        else:
            print(f"    {warn(k):<38} факт: {json.dumps(act, ensure_ascii=False)}  |  ожид: {json.dumps(exp, ensure_ascii=False)}")


def show_tool_call(tc: dict, label: str):
    print(f"\n  [{label}]")
    print(f"    tool : {bold(tc['tool'])}")
    print(f"    args : {json.dumps(tc.get('args', {}), ensure_ascii=False)}")


def compute_f1_verbose(actual_list, expected_list, name: str) -> float:
    if not expected_list:
        return 1.0
    if not actual_list:
        tp = 0
    else:
        tp = len(set(actual_list) & set(expected_list))
    precision = tp / len(actual_list) if actual_list else 0.0
    recall    = tp / len(expected_list)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    print(f"    {name:<16}  TP={tp}  P={precision:.2f}  R={recall:.2f}  F1={f1:.2f}")
    return f1


# ══════════════════════════════════════════════════════════════════════════════

def main():
    sec("WALKTHROUGH: разбор одного запроса")

    print(f"\n  Запрос : {bold(QUERY)}")
    print(f"  Категория : search_summaries  (сложность: medium)")
    print(f"\n  Ключевое слово «что обсуждалось» → ожидается search-summaries,")
    print(f"  а не search. Именно эту ошибку делает старый агент.\n")

    # ── 1. Эталон ─────────────────────────────────────────────────────────────
    sub("1. ЭТАЛОН (golden annotation)")
    exp_tc = GOLDEN["expected_tool_calls"][0]
    print(f"  tool    : {bold(exp_tc['tool'])}")
    print(f"  args    : {json.dumps(exp_tc['args'], ensure_ascii=False, indent=10)}")
    print(f"\n  S_opt = {len(GOLDEN['expected_tool_calls'])}  (количество ожидаемых вызовов)")

    # ── 2. Старый агент ───────────────────────────────────────────────────────
    sub("2. СТАРЫЙ АГЕНТ (без ReAct)")

    old_tc = OLD_TRACE["tool_calls"][0]
    show_tool_call(old_tc, "фактический вызов")

    tool_match_old = old_tc["tool"] == exp_tc["tool"]
    print(f"\n  Tool Selection : {ok('верно') if tool_match_old else err('неверно')}")
    print(f"    ожидалось: {exp_tc['tool']},  получили: {old_tc['tool']}")

    print(f"\n  Аргументы (сравнение):")
    print_args_diff(old_tc.get("args", {}), exp_tc.get("args", {}))

    print(f"\n  A_F1 компоненты:")
    if not tool_match_old:
        print(f"    Инструмент неверный → A_F1 = 0.00 (без проверки аргументов)")
        a_f1_old = 0.0
    else:
        a_f1_old = compute_argument_f1(None, GOLDEN)

    print(f"\n  Trajectory : нет (старый агент не имеет шагов)")
    print(f"  S_act = 1,  recovery = 0,  loops = 0")

    s_opt = 1
    s_act_old = 1
    r_old = 0
    L_old = 0
    T_old = 1.0 if tool_match_old else 0.0
    C_old = min(1.0, s_opt / max(s_act_old - r_old, s_opt))
    U_old = WEIGHTS["alpha"] * T_old + WEIGHTS["beta"] * a_f1_old + WEIGHTS["gamma"] * C_old - WEIGHTS["delta"] * L_old

    print(f"\n  U(τ) = α·T + β·A_F1 + γ·C − δ·L")
    print(f"       = {WEIGHTS['alpha']}·{T_old:.2f} + {WEIGHTS['beta']}·{a_f1_old:.2f} + {WEIGHTS['gamma']}·{C_old:.2f} − {WEIGHTS['delta']}·{L_old}")
    print(f"       = {WEIGHTS['alpha']*T_old:.3f} + {WEIGHTS['beta']*a_f1_old:.3f} + {WEIGHTS['gamma']*C_old:.3f} − {WEIGHTS['delta']*L_old:.3f}")
    print(f"       = {bold(f'{U_old:.3f}')}")

    # ── 3. Новый агент (ReAct) ────────────────────────────────────────────────
    sub("3. НОВЫЙ АГЕНТ (ReAct)")

    steps = NEW_TRACE["steps"]
    print(f"  Шагов в траектории: {len(steps)}\n")

    for i, step in enumerate(steps, 1):
        print(f"  {'─'*56}")
        print(f"  Шаг {i}")
        print(f"  {'─'*56}")
        print(f"  [THOUGHT]")
        # wrap text
        thought = step["thought"]
        words = thought.split()
        line = "    "
        for w in words:
            if len(line) + len(w) > 62:
                print(line)
                line = "    " + w + " "
            else:
                line += w + " "
        if line.strip():
            print(line)

        show_tool_call({"tool": step["tool"], "args": step["args"]}, "ACTION")

        obs = step["observation"]
        summaries = obs.get("summaries", [])
        print(f"\n  [OBSERVATION]")
        if summaries:
            print(f"    total: {obs.get('total', len(summaries))}")
            for s in summaries:
                print(f"    period : {s['period']}")
                print(f"    summary: {s['summary']}")
        else:
            print(f"    (пусто)")

    print(f"\n  [FINAL ANSWER]")
    print(f"  {NEW_TRACE['final_answer']}")

    # ── 4. Метрики нового агента ──────────────────────────────────────────────
    sub("4. МЕТРИКИ нового агента")

    new_tc = steps[0]
    tool_match_new = new_tc["tool"] == exp_tc["tool"]
    print(f"  Tool Selection : {ok('верно') if tool_match_new else err('неверно')}")
    print(f"    ожидалось: {exp_tc['tool']},  получили: {new_tc['tool']}")

    print(f"\n  Аргументы (сравнение):")
    print_args_diff(new_tc.get("args", {}), exp_tc.get("args", {}))

    print(f"\n  A_F1 компоненты:")
    exp_args = exp_tc.get("args", {})
    f1s = []

    # query (binary)
    q_act = new_tc["args"].get("query", "")
    q_exp = exp_args.get("query", "")
    q_ok  = bool(q_act) and bool(q_exp)
    print(f"    {'query':<16}  present={q_ok}  → binary 1.0" if q_ok else f"    {'query':<16}  missing → 0.0")
    f1s.append(1.0 if q_ok else 0.0)

    if exp_args.get("chat_names"):
        f = compute_f1_verbose(new_tc["args"].get("chat_names"), exp_args["chat_names"], "chat_names")
        f1s.append(f)

    if exp_args.get("date_from") is not None:
        d_ok = new_tc["args"].get("date_from") == exp_args["date_from"]
        print(f"    {'date_from':<16}  {'✓' if d_ok else '✗'}  fact={new_tc['args'].get('date_from')}  exp={exp_args['date_from']}")
        f1s.append(1.0 if d_ok else 0.0)

    if exp_args.get("summary_type") is not None:
        s_ok = new_tc["args"].get("summary_type") == exp_args["summary_type"]
        print(f"    {'summary_type':<16}  {'✓' if s_ok else '✗'}  fact={new_tc['args'].get('summary_type')}  exp={exp_args['summary_type']}")
        f1s.append(1.0 if s_ok else 0.0)

    a_f1_new = sum(f1s) / len(f1s) if f1s else 1.0
    print(f"\n  A_F1 = mean({[round(x,2) for x in f1s]}) = {a_f1_new:.3f}")

    s_act_new = len(steps)
    r_new = 0
    L_new = 0
    C_new = min(1.0, s_opt / max(s_act_new - r_new, s_opt))

    print(f"\n  Trajectory:")
    print(f"    S_opt = {s_opt}  (из эталона)")
    print(f"    S_act = {s_act_new}  (фактических шагов)")
    print(f"    R     = {r_new}   (шагов-восстановлений после пустого ответа)")
    print(f"    L     = {L_new}   (повторных одинаковых вызовов)")
    print(f"    C     = min(1, S_opt / max(S_act−R, S_opt))")
    print(f"          = min(1, {s_opt} / max({s_act_new}−{r_new}, {s_opt}))")
    print(f"          = min(1, {s_opt}/{max(s_act_new - r_new, s_opt)}) = {C_new:.3f}")

    T_new = 1.0 if tool_match_new else 0.0
    U_new = WEIGHTS["alpha"] * T_new + WEIGHTS["beta"] * a_f1_new + WEIGHTS["gamma"] * C_new - WEIGHTS["delta"] * L_new

    print(f"\n  U(τ) = α·T + β·A_F1 + γ·C − δ·L")
    print(f"       = {WEIGHTS['alpha']}·{T_new:.2f} + {WEIGHTS['beta']}·{a_f1_new:.2f} + {WEIGHTS['gamma']}·{C_new:.2f} − {WEIGHTS['delta']}·{L_new}")
    print(f"       = {WEIGHTS['alpha']*T_new:.3f} + {WEIGHTS['beta']*a_f1_new:.3f} + {WEIGHTS['gamma']*C_new:.3f} − {WEIGHTS['delta']*L_new:.3f}")
    print(f"       = {bold(f'{U_new:.3f}')}")

    # ── 5. Итог ───────────────────────────────────────────────────────────────
    sub("5. ИТОГОВОЕ СРАВНЕНИЕ")

    max_u = WEIGHTS["alpha"] + WEIGHTS["beta"] + WEIGHTS["gamma"]
    print(f"  {'Метрика':<22} {'Старый':>10} {'Новый':>10}")
    print(f"  {'─'*44}")
    print(f"  {'Tool Selection':<22} {err('search'):>18} {ok('search-summ'):>22}")
    print(f"  {'A_F1':<22} {a_f1_old:>10.3f} {a_f1_new:>10.3f}")
    print(f"  {'C (path conv.)':<22} {C_old:>10.3f} {C_new:>10.3f}")
    print(f"  {'L (loops)':<22} {L_old:>10} {L_new:>10}")
    print(f"  {'─'*44}")
    print(f"  {'U(τ)':<22} {U_old:>10.3f} {U_new:>10.3f}   (max={max_u:.2f})")
    print(f"\n  Δ U(τ) = +{U_new - U_old:.3f}  на одном запросе\n")
    hr2()


if __name__ == "__main__":
    main()
