import gradio as gr
import os
from components.personal_cabinet import resume_component
from components.achievements import achievements_component, achievements_page
from components.ai_consultant import ai_consultant_component

# Чистый бело-голубой стиль
css = """
:root {
    --t1-blue: #0066FF;
    --t1-light-blue: #E6F0FF;
    --t1-white: #FFFFFF;
    --t1-light-gray: #F8FAFC;
    --t1-border: #E2E8F0;
    --t1-text: #2D3748;
}

* {
    color: var(--t1-text) !important;
}

body {
    background: var(--t1-light-gray);
    font-family: 'Inter', sans-serif;
    margin: 0;
    padding: 0;
}

.gradio-container {
    max-width: 1400px !important;
    background: var(--t1-light-gray) !important;
    padding: 20px !important;
}

/* Логотип */
.t1-logo {
    display: flex;
    align-items: center;
    justify-content: center;
}

.t1-logo-icon {
    width: 55px;
    height: 55px;
    background: linear-gradient(135deg, var(--t1-blue) 0%, #0047CC 100%);
    border-radius: 12px;
    display: flex;
    align-items: center;
    justify-content: center;
    color: white !important;
    font-weight: bold;
    font-size: 22px;
    box-shadow: 0 4px 12px rgba(0, 102, 255, 0.3);
}

/* Правая часть */
.t1-header-right {
    display: flex;
    align-items: center;
    gap: 10px;
    white-space: nowrap;
}

/* Разделитель — тоже абсолютный, справа от логотипа */
.t1-divider {
    position: absolute;
    left: 92px;                     /* подбирается под ширину логотипа + gap */
    top: 50%;
    transform: translateY(-50%);
    width: 2px;
    height: 40px;
    background: #E2E8F0;
}

/* Заголовок — абсолютный центр по обеим осям, без переноса */
.t1-title-wrapper {
    position: absolute;
    left: 50%;
    top: 50%;
    transform: translate(-50%, -50%); /* делаем центр текста совпадающим с центром блока */
    display: flex;
    align-items: center;
    gap: 10px;
    white-space: nowrap;            /* принудительно в одну строку */
}

.t1-subtitle {
    font-size: 18px;
    color: #64748B !important;
    font-weight: 500;
    font-family: 'Inter', sans-serif;
}

.t1-main-title {
    font-size: 28px;
    font-weight: 700;
    color: var(--t1-blue) !important;
    font-family: 'Inter', sans-serif;
    letter-spacing: -0.5px;
}

.t1-header {
    background: var(--t1-white);
    border-radius: 12px;
    padding: 15px 25px;
    margin-bottom: 20px;
    box-shadow: 0 2px 8px rgba(0, 102, 255, 0.1);
    display: flex;
    align-items: center;   /* центрируем всё по вертикали */
    justify-content: flex-start; /* логотип уходит в левый край */
    border: 1px solid var(--t1-border);
    gap: 20px;
    white-space: nowrap;   /* запрещаем перенос */
}

.t1-sidebar {
    background: var(--t1-white);
    border-radius: 12px;
    padding: 20px;
    box-shadow: 0 2px 8px rgba(0, 102, 255, 0.1);
    border: 1px solid var(--t1-border);
    height: fit-content;
}

.t1-card {
    background: var(--t1-white);
    border-radius: 12px;
    padding: 25px;
    box-shadow: 0 2px 8px rgba(0, 102, 255, 0.1);
    border: 1px solid var(--t1-border);
    margin-bottom: 20px;
}

.t1-button {
    background: var(--t1-blue) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 12px 24px !important;
    font-weight: 500 !important;
    cursor: pointer !important;
}

.t1-button:hover {
    background: #0055DD !important;
}

.t1-button-secondary {
    background: var(--t1-white) !important;
    color: var(--t1-blue) !important;
    border: 1px solid var(--t1-blue) !important;
    border-radius: 8px !important;
    padding: 10px 20px !important;
    font-weight: 500 !important;
}

.t1-button-secondary:hover {
    background: var(--t1-light-blue) !important;
}

.t1-chat-container {
    border: 1px solid var(--t1-border);
    border-radius: 12px;
    padding: 15px;
    background: var(--t1-white);
    height: 350px;
    display: flex;
    flex-direction: column;
    cursor: pointer;
    transition: all 0.3s ease;
}

.t1-chat-container.expanded {
    position: fixed;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    width: 600px;
    height: 500px;
    z-index: 1000;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
}

.t1-chat-messages {
    flex-grow: 1;
    overflow-y: auto;
    margin-bottom: 15px;
    padding: 10px;
    background: var(--t1-light-gray);
    border-radius: 8px;
}

.t1-chat-input {
    display: flex;
    gap: 10px;
}

.t1-ai-bubble {
    background: var(--t1-light-blue);
    border-radius: 18px 18px 18px 4px;
    padding: 12px 16px;
    margin: 8px 0;
    max-width: 80%;
    align-self: flex-start;
}

.t1-user-bubble {
    background: var(--t1-blue);
    color: white !important;
    border-radius: 18px 18px 4px 18px;
    padding: 12px 16px;
    margin: 8px 0;
    max-width: 80%;
    align-self: flex-end;
}

.t1-user-bubble * {
    color: white !important;
}

.t1-avatar {
    width: 50px;
    height: 50px;
    border-radius: 50%;
    background: var(--t1-blue);
    color: blue !important;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: bold;
    font-size: 18px;
    margin-right: 15px;
}

.t1-progress-container {
    height: 8px;
    background: var(--t1-light-gray);
    border-radius: 4px;
    overflow: hidden;
    margin: 10px 0;
    border: 1px solid var(--t1-border);
}

.t1-progress-bar {
    height: 100%;
    background: var(--t1-blue);
    border-radius: 4px;
}

.t1-menu-item {
    padding: 15px;
    border-radius: 8px;
    margin-bottom: 8px;
    cursor: pointer;
    display: flex;
    align-items: center;
    gap: 12px;
    transition: all 0.2s ease;
    border: 1px solid transparent;
}

.t1-menu-item:hover {
    background: var(--t1-light-blue);
    border-color: var(--t1-blue);
}

.t1-menu-item.active {
    background: var(--t1-blue);
    color: black !important;
    border-color: var(--t1-blue);
}

.t1-menu-item.active * {
    color: black !important;
}

.t1-integration-card {
    border: 1px solid var(--t1-border);
    border-radius: 8px;
    padding: 15px;
    margin-bottom: 12px;
    background: var(--t1-white);
}

.t1-integration-card:hover {
    border-color: var(--t1-blue);
}

.t1-badge {
    background: var(--t1-light-blue);
    color: var(--t1-blue) !important;
    border-radius: 16px;
    padding: 6px 12px;
    font-size: 12px;
    font-weight: 500;
    display: inline-block;
    margin: 4px;
    border: 1px solid var(--t1-blue);
}

/* Стили для полей ввода */
input, textarea, select {
    background: var(--t1-white) !important;
    border: 1px solid var(--t1-border) !important;
    color: var(--t1-text) !important;
    border-radius: 8px !important;
    padding: 12px !important;
}

input::placeholder, textarea::placeholder {
    color: #94A3B8 !important;
}

.gr-box {
    background: var(--t1-white) !important;
    border: 1px solid var(--t1-border) !important;
    border-radius: 8px !important;
}

.overlay {
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: rgba(0, 0, 0, 0.5);
    z-index: 999;
    display: none;
}
/* Стили для полей ввода и текстовых областей */
input[type="text"], input[type="email"], input[type="number"], textarea, select {
    background: var(--t1-white) !important;
    border: 1px solid var(--t1-border) !important;
    color: var(--t1-text) !important;
    border-radius: 8px !important;
    padding: 12px !important;
    margin: 5px 0 !important;
}

/* Фон контейнеров полей ввода */
.gr-box, .gr-input, .gr-textarea {
    background: var(--t1-light-gray) !important;
    border: 1px solid var(--t1-border) !important;
    border-radius: 12px !important;
    padding: 15px !important;
    margin: 10px 0 !important;
}

/* Ховер-эффекты для полей ввода */
input[type="text"]:focus, input[type="email"]:focus, 
input[type="number"]:focus, textarea:focus, select:focus {
    border-color: var(--t1-blue) !important;
    box-shadow: 0 0 0 2px rgba(0, 102, 255, 0.1) !important;
    outline: none !important;
}

/* Стили для лейблов */
label {
    background: transparent !important;
    color: var(--t1-text) !important;
    font-weight: 600 !important;
    margin-bottom: 5px !important;
    padding: 0 !important;
}

/* Контейнеры форм */
.gr-form {
    background: var(--t1-light-gray) !important;
    border-radius: 12px !important;
    padding: 15px !important;
    margin: 10px 0 !important;
    border: 1px solid var(--t1-border) !important;
}

/* Убираем темные фоны у всех элементов */
.gr-block, .gr-group, .gr-compact {
    background: transparent !important;
    border: none !important;
}

/* Специфичные стили для различных компонентов Gradio */
[data-testid="textbox"], [data-testid="number-input"], 
[data-testid="dropdown"], [data-testid="slider"] {
    background: var(--t1-white) !important;
    border-color: var(--t1-border) !important;
}

/* Стили для слайдеров */
.gr-slider > .gr-group {
    background: var(--t1-light-gray) !important;
    padding: 15px !important;
    border-radius: 12px !important;
    border: 1px solid var(--t1-border) !important;
}

/* Стили для файлового загрузчика */
.gr-file {
    background: var(--t1-light-gray) !important;
    border: 1px solid var(--t1-border) !important;
    border-radius: 12px !important;
    padding: 15px !important;
}

/* Убираем темные тени */
.gr-box, .gr-input, .gr-textarea, .gr-file {
    box-shadow: none !important;
}

/* Стили для кнопок внутри форм */
.gr-form .t1-button {
    margin-top: 10px !important;
}

/* Гарантируем, что все текстовые элементы имеют правильный цвет */
.gr-markdown, .gr-text, .gr-label {
    color: var(--t1-text) !important;
    background: transparent !important;
}

/* Стили для чекбоксов и радио кнопок */
.gr-checkbox, .gr-radio {
    background: var(--t1-light-gray) !important;
    border: 1px solid var(--t1-border) !important;
    border-radius: 8px !important;
    padding: 10px !important;
    margin: 5px 0 !important;
}

/* Специфичный селектор для темных областей */
.dark, .gr-panel, .gr-card {
    background: var(--t1-light-gray) !important;
    color: var(--t1-text) !important;
}

/* Убираем любые темные градиенты */
.gr-container, .gr-app {
    background: var(--t1-light-gray) !important;
    background-image: none !important;
}

/* Важный! Переопределяем все возможные темные фоны */
* {
    background-color: transparent !important;
}

/* Но делаем исключение для нужных элементов */
.t1-card, .t1-sidebar, .t1-header, 
input[type="text"], input[type="email"], 
input[type="number"], textarea, select,
.gr-box, .gr-input, .gr-textarea {
    background: var(--t1-white) !important;
}

/* Для светлых контейнеров */
.t1-chat-container, .t1-chat-messages,
.gr-form, .gr-slider > .gr-group,
.gr-file, .gr-checkbox, .gr-radio {
    background: var(--t1-light-gray) !important;
}
"""

# Основной интерфейс
os.environ.pop('HTTP_PROXY', None); os.environ.pop('HTTPS_PROXY', None); os.environ.pop('ALL_PROXY', None);
os.environ.pop('http_proxy', None); os.environ.pop('https_proxy', None); os.environ.pop('all_proxy', None);
os.environ['NO_PROXY'] = '127.0.0.1,localhost'
with gr.Blocks(css=css, title="CareerAI") as demo:
    # Добавляем overlay для затемнения фона
    gr.HTML("""
    <div class="overlay" id="chat-overlay" onclick="closeExpandedChat()"></div>
    <script>
    function expandChat() {
        document.getElementById('chat-container').classList.add('expanded');
        document.getElementById('chat-overlay').style.display = 'block';
    }

    function closeExpandedChat() {
        document.getElementById('chat-container').classList.remove('expanded');
        document.getElementById('chat-overlay').style.display = 'none';
    }
    </script>
    """)

    # Верхняя панель с красивым логотипом
    with gr.Row(elem_classes="t1-header"):
        # Логотип CAI
        gr.HTML("""
            <div class="t1-header">
                <div class="t1-logo-icon">CAI</div>
                <div class="t1-divider"></div>

                <!-- Абсолютно центрируемый заголовок.
                     Его середина встанет точно в середину родительского .t1-header -->
                <div class="t1-title-wrapper">
                    <span class="t1-main-title">CareerAI</span>
                    <span class="t1-subtitle">- Ваш персональный помощник</span>
                </div>
            </div>
            """)

    # Основной контент
    with gr.Row():
        # Боковое меню слева
        with gr.Column(scale=1, min_width=280):
            with gr.Column(elem_classes="t1-sidebar") as sidebar:
                gr.Markdown("### 📋 Меню")
                    
                with gr.Column():
                    btn_resume = gr.Button("📄 Резюме", elem_classes="t1-menu-item")
                    btn_ach = gr.Button("🏆 Достижения", elem_classes="t1-menu-item")
                    btn_progress = gr.Button("📊 Прогресс", elem_classes="t1-menu-item")

                gr.Markdown("---")
                gr.Markdown("### 🔗 Источники данных")

                with gr.Column():
                    gr.HTML("""
                    <div class="t1-integration-card">
                        <div style="display: flex; align-items: center; justify-content: space-between; width: 100%;">
                            <span>GitHub</span>
                            <button class="t1-button-secondary" style="padding: 8px 16px; font-size: 12px;">Подключить</button>
                        </div>
                    </div>
                    <div class="t1-integration-card">
                        <div style="display: flex; align-items: center; justify-content: space-between; width: 100%;">
                            <span>LinkedIn</span>
                            <button class="t1-button-secondary" style="padding: 8px 16px; font-size: 12px;">Подключить</button>
                        </div>
                    </div>
                    <div class="t1-integration-card">
                        <div style="display: flex; align-items: center; justify-content: space-between; width: 100%;">
                            <span>HeadHunter</span>
                            <button class="t1-button-secondary" style="padding: 8px 16px; font-size: 12px;">Подключить</button>
                        </div>
                    </div>
                    """)

        # Центральная часть - Основной контент (Резюме)
        with gr.Column(scale=2) as main_center:
            # Передаем user_id (пока фиксированный, в реальном приложении нужно добавить аутентификацию)
            with gr.Column(visible=True) as center_resume:
                resume_component(user_id=1)
            with gr.Column(visible=False) as center_achievements:
                achievements_page(user_id=1)

        # Правая колонка - Достижения и ИИ-консультант
        with gr.Column(scale=1):
            achievements_component(user_id=1)
            ai_consultant_component(user_id=1)
            # ---- Обработчики меню ----
            def _show_resume():
                return gr.update(visible=True), gr.update(visible=False)
            def _show_achievements():
                return gr.update(visible=False), gr.update(visible=True)
            btn_resume.click(_show_resume, inputs=[], outputs=[center_resume, center_achievements])
            btn_ach.click(_show_achievements, inputs=[], outputs=[center_resume, center_achievements])




if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7861)