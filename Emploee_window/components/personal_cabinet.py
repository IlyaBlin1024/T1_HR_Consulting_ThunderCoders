import gradio as gr
import re
from components.api_client import get_user_data, update_user_data


def resume_component(user_id: int):
    # Получаем данные пользователя с бэкенда
    user_data = get_user_data(user_id) or {}

    # Функция для валидации телефона
    def validate_phone(phone):
        if phone:
            # Удаляем все нецифровые символы
            digits = re.sub(r'\D', '', phone)

            # Форматируем номер телефона
            if len(digits) == 11:
                return f"+7 ({digits[1:4]}) {digits[4:7]}-{digits[7:9]}-{digits[9:]}"
            elif len(digits) == 10:
                return f"+7 ({digits[0:3]}) {digits[3:6]}-{digits[6:8]}-{digits[8:]}"
            else:
                return digits  # Возвращаем только цифры
        return phone

    # Функция для сохранения данных
    def save_resume(full_name, position, email, phone, experience, english_level, location,
                    skills, last_job, work_period, responsibilities, education, specialty,
                    certificates, about):
        # Подготавливаем данные для отправки
        user_data = {
            "full_name": full_name,
            "position": position,
            "email": email,
            "phone": phone,
            "experience_years": float(experience) if experience else 0.0,
            "grade": english_level,  # Используем поле grade для уровня английского
            "department": location,  # Используем поле department для локации
            "resume_text": f"Навыки: {skills}\n\nПоследнее место работы: {last_job}\nПериод: {work_period}\nОбязанности: {responsibilities}\n\nОбразование: {education}\nСпециальность: {specialty}\n\nСертификаты: {certificates}\n\nО себе: {about}"
        }

        # Отправляем на бэкенд
        success = update_user_data(user_id, user_data)
        return "✅ Данные сохранены!" if success else "❌ Ошибка при сохранении"

    with gr.Column(elem_classes="t1-card"):
        gr.Markdown("## 📄 Резюме")

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Личная информация")
                with gr.Group():
                    full_name = gr.Textbox(
                        label="ФИО",
                        value=user_data.get('full_name', ''),
                        placeholder="Введите ваше ФИО",
                        interactive=True
                    )
                    position_input = gr.Textbox(
                        label="Должность",
                        value=user_data.get('position', ''),
                        placeholder="Например: Python Developer",
                        interactive=True
                    )
                    email_input = gr.Textbox(
                        label="Email",
                        value=user_data.get('email', ''),
                        placeholder="your.email@example.com",
                        interactive=True
                    )

                    # Поле телефона с валидацией
                    phone_input = gr.Textbox(
                        label="Телефон",
                        value=user_data.get('phone', ''),
                        placeholder="+7 (999) 123-45-67",
                        interactive=True,
                        max_lines=1
                    )
                    phone_input.change(
                        fn=validate_phone,
                        inputs=phone_input,
                        outputs=phone_input
                    )

            with gr.Column(scale=1):
                gr.Markdown("### Профессиональная информация")
                with gr.Group():
                    experience_input = gr.Number(
                        label="Опыт работы (лет)",
                        value=user_data.get('experience_years', 0),
                        interactive=True
                    )
                    english_input = gr.Textbox(
                        label="Уровень английского",
                        value=user_data.get('grade', ''),
                        placeholder="Например: Intermediate",
                        interactive=True
                    )
                    location_input = gr.Textbox(
                        label="Предпочтительная локация",
                        value=user_data.get('department', ''),
                        placeholder="Например: Москва/Удаленно",
                        interactive=True
                    )

        gr.Markdown("### Навыки")
        with gr.Group():
            skills_input = gr.Textbox(
                label="Ключевые навыки",
                placeholder="Перечислите ваши основные навыки через запятую",
                interactive=True,
                lines=2
            )

        gr.Markdown("### Опыт работы")
        with gr.Group():
            last_job_input = gr.Textbox(
                label="Последнее место работы",
                placeholder="Название компании",
                interactive=True
            )
            work_period_input = gr.Textbox(
                label="Период работы",
                placeholder="Например: 2020-2023",
                interactive=True
            )
            responsibilities_input = gr.Textbox(
                label="Обязанности",
                placeholder="Опишите ваши основные обязанности",
                interactive=True,
                lines=3
            )

        gr.Markdown("### Образование")
        with gr.Group():
            education_input = gr.Textbox(
                label="ВУЗ",
                placeholder="Название учебного заведения",
                interactive=True
            )
            specialty_input = gr.Textbox(
                label="Специальность",
                placeholder="Ваша специальность",
                interactive=True
            )

        gr.Markdown("### Сертификаты")
        with gr.Group():
            certificates_input = gr.Textbox(
                label="Профессиональные сертификаты",
                placeholder="Перечислите ваши сертификаты",
                interactive=True,
                lines=2
            )

        gr.Markdown("### О себе")
        with gr.Group():
            about_input = gr.Textbox(
                label="Дополнительная информация",
                placeholder="Расскажите о себе",
                interactive=True,
                lines=3
            )

        # Кнопки сохранения
        save_status = gr.Textbox(visible=False)
        with gr.Row():
            save_btn = gr.Button("💾 Сохранить резюме", elem_classes="t1-button")
            export_btn = gr.Button("📤 Экспорт в PDF", elem_classes="t1-button-secondary")

        save_btn.click(
            fn=save_resume,
            inputs=[
                full_name, position_input, email_input, phone_input, experience_input,
                english_input, location_input, skills_input, last_job_input,
                work_period_input, responsibilities_input, education_input,
                specialty_input, certificates_input, about_input
            ],
            outputs=save_status
        )