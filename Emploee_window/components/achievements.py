
import gradio as gr
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional, Set
from components.api_client import get_dashboard_data, add_microstep, get_achievements_catalog


# ====================== Утилиты уровней/XP ======================
def _level_stats(total_xp: int) -> Dict[str, int]:
    """Возвращает словарь с полями:
    level, xp_to_next, xp_in_level, next_threshold, prev_threshold, percent_to_next
    """
    level = 1 + (total_xp // 1000)
    prev_threshold = (level - 1) * 1000
    next_threshold = level * 1000
    xp_in_level = max(0, total_xp - prev_threshold)
    xp_to_next = max(0, next_threshold - total_xp)
    percent_to_next = int(min(100, round((xp_in_level / 1000) * 100)))
    return {
        "level": level,
        "xp_to_next": xp_to_next,
        "xp_in_level": xp_in_level,
        "next_threshold": next_threshold,
        "prev_threshold": prev_threshold,
        "percent_to_next": percent_to_next,
    }


def _parse_dt(s: Any) -> Any:
    try:
        if isinstance(s, str):
            # ISO8601
            return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except Exception:
        pass
    return s


def _latest(achievements: List[Dict[str, Any]], n: int = 5) -> List[Dict[str, Any]]:
    arr = achievements or []
    arr = sorted(arr, key=lambda a: _parse_dt(a.get("obtained_at")), reverse=True)
    return arr[:n]


def _format_recent_md(recent: List[Dict[str, Any]]) -> str:
    if not recent:
        return "*Пока нет достижений*"
    lines = []
    for a in recent:
        title = a.get("title") or a.get("code", "—")
        level = a.get("level", "")
        xp = a.get("xp", 0)
        dt = a.get("obtained_at")
        dt_str = ""
        if isinstance(dt, str):
            try:
                dt_str = datetime.fromisoformat(dt.replace("Z", "+00:00")).strftime("%d.%m.%Y")
            except Exception:
                dt_str = ""
        lines.append(f"- **{title}** — {level} (+{xp} XP){(' · ' + dt_str) if dt_str else ''}")
    return "\n".join(lines)


# ====================== Компонент справа (сайдбар) ======================
def achievements_component(user_id: int):
    data = get_dashboard_data(user_id) or {}
    total_xp = int(data.get("total_xp", 0) or 0)
    ach = data.get("achievements", []) or []

    stats = _level_stats(total_xp)

    with gr.Column(elem_classes="t1-card"):
        gr.Markdown("## 🏆 Достижения")

        # Верхние метрики
        xp_md = gr.Markdown(f"**XP:** {total_xp}")
        level_md = gr.Markdown(f"**Уровень:** {stats['level']}")
        xp_next_md = gr.Markdown(f"**До след. уровня:** {stats['xp_to_next']} XP")

        # Прогресс бар уровня (XP -> следующий уровень)
        prog_html = gr.HTML(f"""
        <div class="t1-progress-container">
            <div class="t1-progress-bar" style="width: {stats['percent_to_next']}%"></div>
        </div>
        """)

        # Кнопка ежедневного шага (стрик)
        daily_btn = gr.Button("✅ Отметить ежедневный прогресс", elem_classes="t1-button")

        # Последние достижения
        gr.Markdown("### Последние достижения")
        recent_md = gr.Markdown(_format_recent_md(_latest(ach, 5)))

        def _refresh():
            d = get_dashboard_data(user_id) or {}
            total = int(d.get("total_xp", 0) or 0)
            st = _level_stats(total)
            a = d.get("achievements", []) or []
            return (
                f"**XP:** {total}",
                f"**Уровень:** {st['level']}",
                f"**До след. уровня:** {st['xp_to_next']} XP",
                f'<div class="t1-progress-container"><div class="t1-progress-bar" style="width: {st["percent_to_next"]}%"></div></div>',
                _format_recent_md(_latest(a, 5)),
            )

        def _do_daily():
            add_microstep(user_id)
            return _refresh()

        daily_btn.click(
            fn=_do_daily,
            inputs=[],
            outputs=[xp_md, level_md, xp_next_md, prog_html, recent_md],
            show_progress=True
        )


# ====================== Полная страница достижений (центр) ======================
def _iter_catalog_levels(catalog: Dict[str, Dict[str, Any]]) -> List[Tuple[str, str, str, int]]:
    """Разворачивает каталог в список кортежей: (code, title, level_label, xp)"""
    rows: List[Tuple[str, str, str, int]] = []
    for code, meta in (catalog or {}).items():
        title = meta.get("title", code)
        if "levels" in meta:
            for lvl in meta["levels"]:
                # варианты: ("бронза", 30, 0.40) или ("B2", 80)
                if isinstance(lvl, (list, tuple)) and len(lvl) >= 2:
                    level_label = str(lvl[0])
                    xp = int(lvl[1])
                    rows.append((code, title, level_label, xp))
        if "thresholds" in meta:
            for t, xp in meta["thresholds"]:
                t = int(t)
                if code == "availability":
                    level_label = f"{t}m+"
                elif code == "compliance":
                    level_label = f"step{t}"
                else:
                    level_label = f"{t}+"
                rows.append((code, title, level_label, int(xp)))
        if "absolute" in meta:
            abs_level, abs_xp = meta["absolute"]
            rows.append((code, title, str(abs_level), int(abs_xp)))
    return rows


def _split_done_vs_locked(all_levels: List[Tuple[str, str, str, int]], achieved: List[Dict[str, Any]]):
    got_set: Set[Tuple[str, str]] = {(a.get("code"), a.get("level")) for a in (achieved or [])}
    done = [r for r in all_levels if (r[0], r[2]) in got_set]
    locked = [r for r in all_levels if (r[0], r[2]) not in got_set]
    return done, locked


def achievements_page(user_id: int):
    data = get_dashboard_data(user_id) or {}
    total_xp = int(data.get("total_xp", 0) or 0)
    ach = data.get("achievements", []) or []
    catalog = get_achievements_catalog() or {}

    stats = _level_stats(total_xp)
    all_rows = _iter_catalog_levels(catalog)
    done, locked = _split_done_vs_locked(all_rows, ach)

    with gr.Column(elem_classes="t1-card"):
        gr.Markdown("## 🏆 Все достижения")
        gr.Markdown(f"**Уровень:** {stats['level']} &nbsp;&nbsp;•&nbsp;&nbsp; **XP:** {total_xp} &nbsp;&nbsp;•&nbsp;&nbsp; **До следующего:** {stats['xp_to_next']} XP")
        gr.HTML(f"""
        <div class="t1-progress-container">
            <div class="t1-progress-bar" style="width: {stats['percent_to_next']}%"></div>
        </div>
        """)

        with gr.Tab("Выполненные"):
            if done:
                for code, title, level_label, xp in sorted(done, key=lambda x: (x[0], x[2])):
                    gr.Markdown(f"✅ **{title}** — {level_label} (+{xp} XP)")
            else:
                gr.Markdown("*Пока нет выполненных достижений*")

        with gr.Tab("Невыполненные"):
            if locked:
                for code, title, level_label, xp in sorted(locked, key=lambda x: (x[0], x[2])):
                    gr.Markdown(f"⬜ **{title}** — {level_label} (+{xp} XP)")
            else:
                gr.Markdown("🎉 Все уровни достижений закрыты!")
