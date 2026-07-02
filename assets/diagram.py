"""
Генерирует PNG-схему evaluation pipeline для презентации.
Запуск: python diagram.py
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

fig, ax = plt.subplots(figsize=(14, 10))
ax.set_xlim(0, 14)
ax.set_ylim(0, 10)
ax.axis("off")
fig.patch.set_facecolor("#FFFFFF")
ax.set_facecolor("#FFFFFF")

# ── Цвета ─────────────────────────────────────────────────────────────────────
C_BG      = "#FFFFFF"
C_INPUT   = "#EBF3FB"
C_SCRIPT  = "#D6E8F7"
C_DATA    = "#D6F0E0"
C_EVAL    = "#E8D6F7"
C_BORDER  = "#2980B9"
C_GREEN   = "#1A7A40"
C_PURPLE  = "#6C3483"
C_ORANGE  = "#B7470A"
C_RED     = "#C0392B"
C_TEXT    = "#1A1A2E"
C_SUBTLE  = "#555555"
C_ARROW   = "#2980B9"


def box(ax, x, y, w, h, label, sublabel="", color=C_INPUT, border=C_BORDER,
        fontsize=11, subfontsize=9, radius=0.3):
    rect = FancyBboxPatch(
        (x - w/2, y - h/2), w, h,
        boxstyle=f"round,pad=0.05,rounding_size={radius}",
        facecolor=color, edgecolor=border, linewidth=1.5, zorder=3
    )
    ax.add_patch(rect)
    ax.text(x, y + (0.15 if sublabel else 0), label,
            ha="center", va="center", color=C_TEXT,
            fontsize=fontsize, fontweight="bold", zorder=4)
    if sublabel:
        ax.text(x, y - 0.28, sublabel,
                ha="center", va="center", color=C_SUBTLE,
                fontsize=subfontsize, zorder=4)


def arrow(ax, x1, y1, x2, y2, color=C_ARROW):
    ax.annotate("",
        xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(
            arrowstyle="-|>", color=color,
            lw=1.8, mutation_scale=16,
        ), zorder=5
    )


def label_on_arrow(ax, x, y, text, color=C_SUBTLE, fontsize=8):
    ax.text(x, y, text, ha="center", va="center",
            color=color, fontsize=fontsize,
            bbox=dict(facecolor=C_BG, edgecolor="none", pad=1.5), zorder=6)


# ── Заголовок ─────────────────────────────────────────────────────────────────
ax.text(7, 9.55, "Evaluation Pipeline",
        ha="center", va="center", color=C_TEXT,
        fontsize=15, fontweight="bold")
ax.text(7, 9.15, "Как оценить нового ReAct-агента без ground truth",
        ha="center", va="center", color=C_SUBTLE, fontsize=10)

# ── Верхний блок: 1000 запросов ───────────────────────────────────────────────
box(ax, 7, 8.3, 4.2, 0.85,
    "1 000 реальных запросов пользователей",
    "user_queries_public.json",
    color=C_INPUT, border="#4A90D9", fontsize=11)

# ── Развилка вниз-влево и вниз-вправо ─────────────────────────────────────────
# Линия вниз от запросов
ax.plot([7, 7], [7.87, 7.5], color=C_ARROW, lw=1.8, zorder=5)
# Горизонтальная развилка
ax.plot([3.5, 10.5], [7.5, 7.5], color=C_ARROW, lw=1.8, zorder=5)
# Стрелки вниз
arrow(ax, 3.5, 7.5, 3.5, 6.85)
arrow(ax, 10.5, 7.5, 10.5, 6.85)

label_on_arrow(ax, 1.9, 7.5, "авто-разметка", color="#4A90D9", fontsize=8)
label_on_arrow(ax, 12.1, 7.5, "прогон агента", color="#4A90D9", fontsize=8)

# ── Левая ветка: annotate.py ──────────────────────────────────────────────────
box(ax, 3.5, 6.4, 3.8, 0.8,
    "annotate.py",
    "LLM авто-разметка запросов",
    color=C_SCRIPT, border="#4A90D9", fontsize=11)

arrow(ax, 3.5, 6.0, 3.5, 5.35)
label_on_arrow(ax, 2.2, 5.7, "для каждого запроса\n→ ожидаемый tool call", fontsize=7.5)

box(ax, 3.5, 4.9, 3.8, 0.8,
    "golden_dataset.json",
    "1000 эталонов: tool + args",
    color=C_DATA, border=C_GREEN, fontsize=11)

# ── Правая ветка: новый агент ─────────────────────────────────────────────────
box(ax, 10.5, 6.4, 3.8, 0.8,
    "Новый ReAct-агент",
    "запускается на тех же запросах",
    color=C_SCRIPT, border=C_PURPLE, fontsize=11)

arrow(ax, 10.5, 6.0, 10.5, 5.35)
label_on_arrow(ax, 12.2, 5.7, "для каждого запроса\n→ шаги + ответ", fontsize=7.5)

box(ax, 10.5, 4.9, 3.8, 0.8,
    "agent_traces.json",
    "steps: thought → tool → observation",
    color=C_DATA, border=C_PURPLE, fontsize=11)

# ── Стрелки вниз от обоих к evaluate.py ──────────────────────────────────────
arrow(ax, 3.5, 4.5, 3.5, 3.9)
arrow(ax, 10.5, 4.5, 10.5, 3.9)
# Горизонтальные линии к центру
ax.plot([3.5, 7], [3.9, 3.9], color=C_ARROW, lw=1.8, zorder=5)
ax.plot([10.5, 7], [3.9, 3.9], color=C_ARROW, lw=1.8, zorder=5)
arrow(ax, 7, 3.9, 7, 3.35)

# ── evaluate.py ───────────────────────────────────────────────────────────────
box(ax, 7, 2.95, 4.2, 0.75,
    "evaluate.py",
    "--alpha 0.4  --beta 0.3  --gamma 0.3  --delta 0.5",
    color=C_EVAL, border=C_PURPLE, fontsize=12)

arrow(ax, 7, 2.57, 7, 1.95)

# ── Отчёт ─────────────────────────────────────────────────────────────────────
report_x, report_y = 7, 1.1
rect_report = FancyBboxPatch(
    (3.2, 0.15), 7.6, 1.7,
    boxstyle="round,pad=0.05,rounding_size=0.3",
    facecolor="#FFF8F0", edgecolor=C_ORANGE, linewidth=2, zorder=3
)
ax.add_patch(rect_report)

ax.text(7, 1.7, "Отчёт с метриками", ha="center", va="center",
        color=C_TEXT, fontsize=11, fontweight="bold", zorder=4)

metrics = [
    ("Tool Selection", "70%", "97%", "+27%"),
    ("Security Refusal", " 0%", "100%", "+100%"),
    ("A_F1 (args)",    "0.65", "0.97", "+0.32"),
    ("U(τ) avg",       "0.79", "0.99", "+0.20"),
]

cols = [4.0, 6.2, 8.4, 10.2]
ax.text(cols[0], 1.3, "Метрика",      ha="left",   color=C_SUBTLE, fontsize=8.5, zorder=4)
ax.text(cols[1], 1.3, "Старый",       ha="center", color=C_SUBTLE, fontsize=8.5, zorder=4)
ax.text(cols[2], 1.3, "Новый",        ha="center", color=C_SUBTLE, fontsize=8.5, zorder=4)
ax.text(cols[3], 1.3, "Δ",            ha="center", color=C_SUBTLE, fontsize=8.5, zorder=4)

for i, (name, old, new, delta) in enumerate(metrics):
    row_y = 1.0 - i * 0.22
    ax.text(cols[0], row_y, name,  ha="left",   color=C_TEXT,   fontsize=8.5, zorder=4)
    ax.text(cols[1], row_y, old,   ha="center", color="#E74C3C", fontsize=8.5, zorder=4)
    ax.text(cols[2], row_y, new,   ha="center", color=C_GREEN,  fontsize=8.5, zorder=4)
    ax.text(cols[3], row_y, delta, ha="center", color=C_ORANGE, fontsize=8.5,
            fontweight="bold", zorder=4)

# ── Легенда внизу ─────────────────────────────────────────────────────────────
legend_items = [
    (C_SCRIPT, "Скрипт"),
    (C_DATA,   "Данные"),
    (C_EVAL,   "Оценка"),
]
lx = 0.5
for color, label in legend_items:
    rect = FancyBboxPatch((lx, 0.0), 0.3, 0.18,
                           boxstyle="round,pad=0.02",
                           facecolor=color, edgecolor=C_BORDER, lw=0.8, zorder=4)
    ax.add_patch(rect)
    ax.text(lx + 0.45, 0.09, label, va="center", color=C_SUBTLE, fontsize=8, zorder=5)
    lx += 1.5

plt.tight_layout(pad=0.3)
plt.savefig("assets/pipeline_diagram.png", dpi=180, bbox_inches="tight",
            facecolor=C_BG, edgecolor="none")
print("Сохранено: assets/pipeline_diagram.png")
