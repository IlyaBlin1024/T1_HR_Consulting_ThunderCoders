import gradio as gr
from components.api_client import ai_chat


def ai_consultant_component(user_id: int):
    # Функция для обработки сообщений
    
    def respond(message, messages):
        if not message.strip():
            return "", messages

        # фиксируем сообщение пользователя
        messages = (messages or []) + [{"role": "user", "content": message}]
        # Отправляем сообщение на бэкенд
        response = ai_chat(user_id, message) or {}
        bot_message = response.get('reply', 'Не удалось получить ответ')
        # добавляем ответ ассистента
        messages.append({"role": "assistant", "content": bot_message})
        return "", messages

    with gr.Column():
        # Красивый заголовок для ИИ-консультанта
        with gr.Row():
            gr.HTML("""
            <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 15px;">
                <div style="width: 32px; height: 32px; background: linear-gradient(135deg, #0066FF 0%, #0047CC 100%); 
                    border-radius: 8px; display: flex; align-items: center; justify-content: center; color: white; 
                    font-weight: bold; font-size: 16px;">🤖</div>
                <h3 style="margin: 0; font-weight: 600; color: #2D3748;">ИИ-консультант</h3>
            </div>
            """)

        # Чат с ИИ
        with gr.Column(elem_classes="t1-chat-container") as chat_container:
            chat_container.elem_id = "chat-container"

            # Начинаем с приветственного сообщения
            chatbot = gr.Chatbot(type='messages', 
                label="",
                value=[
                    {"role": "assistant", "content": "👋 Здравствуйте! Я ваш ИИ-помощник по карьере. Задайте мне вопрос"}
                ],
                height=250,
                show_label=False,
                elem_classes="t1-chat-messages"
            )

            with gr.Row(elem_classes="t1-chat-input"):
                msg = gr.Textbox(
                    placeholder="Задайте вопрос...",
                    show_label=False,
                    lines=1,
                    container=False,
                    max_lines=1
                )
                send_btn = gr.Button("➤", elem_classes="t1-button", size="sm")

        # Обработчики событий
        msg.submit(respond, [msg, chatbot], [msg, chatbot])
        send_btn.click(respond, [msg, chatbot], [msg, chatbot])