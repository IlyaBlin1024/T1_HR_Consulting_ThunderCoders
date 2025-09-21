
import os
import json
from typing import List, Tuple, Optional

import requests
import gradio as gr
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import importlib, sys
# --- RAG globals ---
RAG_CHAIN = None
RAG_ERR = None


# --- Environment safety for corporate proxies / localhost ---
os.environ.setdefault("NO_PROXY", "127.0.0.1,localhost")
os.environ.setdefault("no_proxy", "127.0.0.1,localhost")
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "false")
os.environ.setdefault("GRADIO_LAUNCH_IN_BROWSER", "false")

# ---------------------------- Config ----------------------------
DEFAULT_BACKEND_URL = os.getenv("HR_BACKEND_URL", "http://localhost:8000")
DEFAULT_HR_USER = os.getenv("HR_USER", "hr1")

# ---------------------------- HTTP helpers ----------------------------
def _post(url: str, json_payload: dict = None, params: dict = None):
    try:
        resp = requests.post(url, json=json_payload, params=params, timeout=30)
        resp.raise_for_status()
        return resp.json(), None
    except Exception as e:
        return None, f"Ошибка запроса: {e}"

def _get(url: str, params: dict = None):
    try:
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        return resp.json(), None
    except Exception as e:
        return None, f"Ошибка запроса: {e}"


# ---------------------------- RAG helpers ----------------------------
def _ensure_rag():
    """Lazily import rag.py and build a chain; cache it globally."""
    global RAG_CHAIN, RAG_ERR
    if RAG_CHAIN is not None or RAG_ERR is not None:
        return RAG_CHAIN, RAG_ERR
    try:
        # Make sure /mnt/data (or current dir) is on sys.path
        here = os.path.dirname(os.path.abspath(__file__))
        if here not in sys.path:
            sys.path.append(here)
        rag = importlib.import_module("rag")
        # Allow model override via env
        model_name = os.getenv("RAG_MODEL", "Qwen2.5-72B-Instruct-AWQ")
        data = rag.load_data()
        docs = rag.create_documents(data)
        RAG_CHAIN = rag.create_rag_chain(docs, model_name=model_name)
        return RAG_CHAIN, None
    except Exception as e:
        RAG_ERR = f"RAG недоступен: {e}"
        return None, RAG_ERR

def rag_answer(question: str) -> tuple[str, str]:
    """Return (answer_markdown, error_text)."""
    chain, err = _ensure_rag()
    if err:
        return "", err
    q = (question or "").strip()
    if not q:
        return "", "Введите вопрос для RAG."
    try:
        ans = chain.invoke(q)
        # LangChain may return an object; cast to string
        return f"### RAG-ответ\n\n{str(ans)}", ""
    except Exception as e:
        return "", f"Ошибка RAG: {e}"

# ---------------------------- Formatting ----------------------------
def _profiles_to_df(items: List[dict]) -> pd.DataFrame:
    if not items:
        return pd.DataFrame(columns=["profile_id", "Имя", "Должность", "Match, %", "Стек/пересечение"])
    rows = []
    for it in items:
        # for /saved we may receive full profiles, normalize keys
        pid = it.get("profile_id") or it.get("id")
        name = it.get("name")
        title = it.get("title")
        match_score = it.get("match_score", None)
        overlap = it.get("skills_overlap", [])
        stack = it.get("stack", [])
        overlap_txt = ", ".join(overlap) if overlap else (", ".join(stack) if stack else "—")
        rows.append({
            "profile_id": pid,
            "Имя": name,
            "Должность": title,
            "Match, %": match_score,
            "Стек/пересечение": overlap_txt
        })
    df = pd.DataFrame(rows)
    # Sort by match if column exists
    if "Match, %" in df.columns and df["Match, %"].notna().any():
        df = df.sort_values(by="Match, %", ascending=False, na_position="last").reset_index(drop=True)
    return df

def _options_from_df(df: pd.DataFrame) -> List[Tuple[str, str]]:
    opts = []
    for _, row in df.iterrows():
        label = f"{row.get('Имя', '—')} — {row.get('Должность', '—')} (id: {row.get('profile_id', '—')})"
        opts.append((label, row.get("profile_id")))
    return opts

def _parse_stack(text: str) -> Optional[List[str]]:
    if not text or not text.strip():
        return None
    return [t.strip() for t in text.split(",") if t.strip()]

# ---------------------------- Backend bindings ----------------------------
def do_search(query: str, stack_text: str, min_exp: float, max_salary: str, top_k: int,
              backend_url: str) -> Tuple[pd.DataFrame, List[Tuple[str, str]], str]:
    payload = {
        "direction": (query or None),
        "min_experience": (None if min_exp is None else float(min_exp)),
        "stack": _parse_stack(stack_text),
        "max_salary": (None if not max_salary else int(max_salary)),
        "top_k": int(top_k) if top_k else 5
    }
    data, err = _post(f"{backend_url}/search_employees/", json_payload=payload)
    if err:
        return pd.DataFrame(), [], err
    df = _profiles_to_df(data)
    return df, _options_from_df(df), ""

def do_search_saved(hr_user: str, query: str, stack_text: str, min_exp: float, max_salary: str,
                    top_k: int, backend_url: str) -> Tuple[pd.DataFrame, List[Tuple[str, str]], str]:
    payload = {
        "direction": (query or None),
        "min_experience": (None if min_exp is None else float(min_exp)),
        "stack": _parse_stack(stack_text),
        "max_salary": (None if not max_salary else int(max_salary)),
        "top_k": int(top_k) if top_k else 5
    }
    params = {"hr_user": hr_user or DEFAULT_HR_USER}
    data, err = _post(f"{backend_url}/search_saved_employees/", json_payload=payload, params=params)
    if err:
        return pd.DataFrame(), [], err
    df = _profiles_to_df(data)
    return df, _options_from_df(df), ""

def do_saved_list(hr_user: str, backend_url: str) -> Tuple[pd.DataFrame, List[Tuple[str, str]], str]:
    data, err = _get(f"{backend_url}/saved/{hr_user}")
    if err:
        return pd.DataFrame(), [], err
    df = _profiles_to_df(data)
    return df, _options_from_df(df), ""

def save_for_later(hr_user: str, profile_id: str, backend_url: str) -> str:
    if not profile_id:
        return "Сначала выберите сотрудника в выпадающем списке."
    _, err = _post(f"{backend_url}/save_for_later/{hr_user}/{profile_id}")
    return "Сохранено!" if not err else err

def get_profile_report(profile_id: str, backend_url: str):
    return _get(f"{backend_url}/employee_report/{profile_id}")

# ---------------------------- Charts ----------------------------
def plot_skill_radar(skills: List[dict]):
    # top 5 by level
    if not skills:
        fig, ax = plt.subplots()
        ax.set_title("Нет данных по навыкам")
        return fig
    top = sorted(skills, key=lambda s: float(s.get("level", 0)), reverse=True)[:5]
    labels = [s.get("name", "—") for s in top]
    values = [float(s.get("level", 0)) for s in top]

    # Radar chart
    N = len(values)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    values += values[:1]
    angles += angles[:1]

    fig = plt.figure()
    ax = fig.add_subplot(111, polar=True)
    ax.plot(angles, values, linewidth=2)
    ax.fill(angles, values, alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_yticklabels([])
    ax.set_title("Навыки (топ-5)", va='bottom')
    return fig

def plot_overall_forecast(points: List[float]):
    fig, ax = plt.subplots()
    if not points:
        ax.set_title("Нет прогноза компетенций")
        return fig
    months = [0, 3, 6, 9, 12][:len(points)]
    ax.plot(months, points, marker="o")
    ax.set_xlabel("Месяцы")
    ax.set_ylabel("Средний уровень навыков")
    ax.set_title("Прогноз развития компетенций")
    ax.grid(True, alpha=0.3)
    return fig

def plot_potential_gauge(value: float):
    # Simple horizontal bar-like gauge
    fig, ax = plt.subplots()
    ax.barh([0], [value])
    ax.set_xlim(0, 100)
    ax.set_yticks([])
    ax.set_xlabel("Кадровый потенциал")
    ax.set_title(f"Потенциал: {value:.1f}")
    return fig

# ---------------------------- UI callbacks ----------------------------
def open_profile(profile_id: str, backend_url: str):
    if not profile_id:
        return "Выберите сотрудника.", None, None, None
    report, err = get_profile_report(profile_id, backend_url)
    if err or not report:
        return err or "Ошибка загрузки отчёта", None, None, None

    # Text summary
    name = report.get("name") or "—"
    title = report.get("title") or "—"
    stack = ", ".join(report.get("stack", []) or [])
    years = report.get("years_experience", "—")
    desired_salary = report.get("desired_salary", "—")
    summary = report.get("summary") or "—"
    md = f"### {name}\n**{title}**\n\n**Опыт:** {years} лет\n\n**Стек:** {stack or '—'}\n\n**Ожидания по ЗП:** {desired_salary}\n\n{summary}"

    # Data for tables and charts
    skills = report.get("skills", [])
    skills_rows = []
    for s in skills:
        skills_rows.append({
            "Навык": s.get("name", "—"),
            "Уровень": s.get("level", "—"),
            "Лет опыта": s.get("years", "—"),
            "Категория": s.get("category", "—"),
            "Ключевые слова": ", ".join(s.get("keywords", []) or [])
        })
    skills_df = pd.DataFrame(skills_rows) if skills_rows else pd.DataFrame(
        columns=["Навык", "Уровень", "Лет опыта", "Категория", "Ключевые слова"]
    )

    forecast_fig = plot_overall_forecast(report.get("overall_skill_forecast", []))
    radar_fig = plot_skill_radar(skills)
    potential = float(report.get("hr_potential", 0.0))
    potential_fig = plot_potential_gauge(potential)

    return md, skills_df, radar_fig, forecast_fig, potential_fig

# ---------------------------- Build UI ----------------------------
with gr.Blocks(title="HR-страница: поиск и профиль сотрудника") as demo:
    gr.Markdown("# Поиск сотрудника")
    with gr.Row():
        with gr.Column(scale=3):
            backend_url = gr.Textbox(label="Backend URL", value=DEFAULT_BACKEND_URL, info="Адрес сервера FastAPI")
            hr_user = gr.Textbox(label="HR-пользователь", value=DEFAULT_HR_USER)
            query = gr.Textbox(label="Поисковая строка", placeholder="Например: product manager, аналитик, devops...")
            stack = gr.Textbox(label="Стек (через запятую)", placeholder="Python, Kafka, PostgreSQL")
            with gr.Row():
                min_exp = gr.Slider(label="Минимальный опыт (лет)", value=0.0, minimum=0.0, maximum=30.0, step=0.5)
                top_k = gr.Slider(label="Top K", value=5, minimum=1, maximum=50, step=1)
            max_salary = gr.Textbox(label="Максимальная зарплата (необязательно)", placeholder="например 200000")
            btn_search = gr.Button("Фильтры ▶︎ Найти")
            btn_rag = gr.Button("RAG ▶︎ Подсказать по запросу")
            search_error = gr.Markdown(visible=False)
            rag_md = gr.Markdown(visible=True)

            results_df = gr.Dataframe(headers=["profile_id", "Имя", "Должность", "Match, %", "Стек/пересечение"],
                                      interactive=False, )
            selected_from_search = gr.Dropdown(label="Выберите сотрудника из результатов", choices=[])
            with gr.Row():
                btn_open = gr.Button("Открыть профиль")
                btn_save = gr.Button("Отложить")

        with gr.Column(scale=2):
            gr.Markdown("## Отложенные сотрудники")
            btn_refresh_saved = gr.Button("Обновить список")
            saved_df = gr.Dataframe(headers=["profile_id", "Имя", "Должность", "Match, %", "Стек/пересечение"],
                                    interactive=False, )
            selected_saved = gr.Dropdown(label="Выберите отложенного сотрудника", choices=[])
            with gr.Row():
                # Search within saved
                saved_query = gr.Textbox(label="Поиск среди отложенных", placeholder="например: analyst")
                saved_stack = gr.Textbox(label="Стек (отложенные, через запятую)", placeholder="Python, SQL")
            with gr.Row():
                saved_min_exp = gr.Slider(label="Мин. опыт (отложенные)", value=0.0, minimum=0.0, maximum=30.0, step=0.5)
                saved_top_k = gr.Slider(label="Top K (отложенные)", value=5, minimum=1, maximum=50, step=1)
            saved_max_salary = gr.Textbox(label="Макс. зарплата (отложенные)", placeholder="например 200000")
            btn_search_saved = gr.Button("Фильтровать отложенных")
            saved_error = gr.Markdown(visible=False)

    gr.Markdown("---")
    gr.Markdown("## Профиль сотрудника")
    profile_md = gr.Markdown("Выберите сотрудника слева и нажмите «Открыть профиль».")
    skills_table = gr.Dataframe(interactive=False, )
    radar_plot = gr.Plot()
    forecast_plot = gr.Plot()
    potential_plot = gr.Plot()

    # -------- Wire events --------
    def _on_rag(query):
        md, err = rag_answer(query)
        return gr.update(value=(f"⚠️ {err}" if err else ""), visible=bool(err)), gr.update(value=md or "Нет ответа.")

    btn_rag.click(
        _on_rag,
        inputs=[query],
        outputs=[search_error, rag_md]
    )

    def _on_search(query, stack, min_exp, max_salary, top_k, backend_url):
        df, options, err = do_search(query, stack, min_exp, max_salary, top_k, backend_url)
        return (
            gr.update(value=df),
            gr.update(choices=options, value=(options[0][1] if options else None)),
            gr.update(value=(f"⚠️ {err}" if err else ""), visible=bool(err))
        )

    btn_search.click(
        _on_search,
        inputs=[query, stack, min_exp, max_salary, top_k, backend_url],
        outputs=[results_df, selected_from_search, search_error]
    )

    def _on_refresh_saved(hr_user, backend_url):
        df, options, err = do_saved_list(hr_user, backend_url)
        return (
            gr.update(value=df),
            gr.update(choices=options, value=(options[0][1] if options else None)),
            gr.update(value=(f"⚠️ {err}" if err else ""), visible=bool(err))
        )

    btn_refresh_saved.click(
        _on_refresh_saved,
        inputs=[hr_user, backend_url],
        outputs=[saved_df, selected_saved, saved_error]
    )

    def _on_search_saved(hr_user, saved_query, saved_stack, saved_min_exp, saved_max_salary, saved_top_k, backend_url):
        df, options, err = do_search_saved(hr_user, saved_query, saved_stack, saved_min_exp, saved_max_salary, saved_top_k, backend_url)
        return (
            gr.update(value=df),
            gr.update(choices=options, value=(options[0][1] if options else None)),
            gr.update(value=(f"⚠️ {err}" if err else ""), visible=bool(err))
        )

    btn_search_saved.click(
        _on_search_saved,
        inputs=[hr_user, saved_query, saved_stack, saved_min_exp, saved_max_salary, saved_top_k, backend_url],
        outputs=[saved_df, selected_saved, saved_error]
    )

    def _on_open(profile_id, backend_url):
        md, table, radar, forecast, potential = open_profile(profile_id, backend_url)
        return md, table, radar, forecast, potential

    btn_open.click(
        _on_open,
        inputs=[selected_from_search, backend_url],
        outputs=[profile_md, skills_table, radar_plot, forecast_plot, potential_plot]
    )

    def _on_open_saved(profile_id, backend_url):
        md, table, radar, forecast, potential = open_profile(profile_id, backend_url)
        return md, table, radar, forecast, potential

    selected_saved.change(
        _on_open_saved,
        inputs=[selected_saved, backend_url],
        outputs=[profile_md, skills_table, radar_plot, forecast_plot, potential_plot]
    )

    def _on_save(hr_user, profile_id, backend_url):
        msg = save_for_later(hr_user, profile_id, backend_url)
        # refresh saved automatically
        df, options, err = do_saved_list(hr_user, backend_url)
        return (
            gr.update(value=f"**{msg}**" if msg else ""),
            gr.update(value=df),
            gr.update(choices=options, value=(options[0][1] if options else None)),
            gr.update(value=(f"⚠️ {err}" if err else ""), visible=bool(err))
        )

    btn_save.click(
        _on_save,
        inputs=[hr_user, selected_from_search, backend_url],
        outputs=[search_error, saved_df, selected_saved, saved_error]
    )

# Allow running with: python hr_gradio_app.py
if __name__ == "__main__":
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "7860"))
    demo.launch(server_name=host, server_port=port, share=False, inbrowser=False)
