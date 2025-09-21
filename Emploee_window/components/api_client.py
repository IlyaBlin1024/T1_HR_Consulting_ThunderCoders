# api_client.py
import requests
from typing import Optional, Dict, Any

BASE_URL = "http://127.0.0.1:8000"
PROXIES = {"http": None, "https": None}


def get_user_data(user_id: int) -> Optional[Dict[str, Any]]:
    try:
        response = requests.get(f"{BASE_URL}/users/{user_id}", timeout=5, proxies=PROXIES)
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 404:
            # Создаем нового пользователя, если не найден
            user_data = {
                "email": f"user{user_id}@example.com",
                "full_name": f"User {user_id}",
                "skills": [],
                "projects": [],
                "certificates": []
            }
            create_response = requests.post(f"{BASE_URL}/users", json=user_data, timeout=5, proxies=PROXIES)
            if create_response.status_code == 200:
                return create_response.json()
        print(f"Ошибка API: {response.status_code} - {response.text}")
    except requests.exceptions.RequestException as e:
        print(f"Ошибка соединения: {e}")
    return None


def update_user_data(user_id: int, user_data: dict) -> bool:
    try:
        response = requests.put(f"{BASE_URL}/users/{user_id}", json=user_data, timeout=5)
        return response.status_code == 200
    except requests.exceptions.RequestException as e:
        print(f"Ошибка соединения: {e}")
        return False


def get_dashboard_data(user_id: int) -> Optional[Dict[str, Any]]:
    try:
        response = requests.get(f"{BASE_URL}/users/{user_id}/dashboard", timeout=5, proxies=PROXIES)
        if response.status_code == 200:
            return response.json()
        print(f"Ошибка API: {response.status_code} - {response.text}")
    except requests.exceptions.RequestException as e:
        print(f"Ошибка соединения: {e}")
    return None


def ai_chat(user_id: int, message: str) -> Dict[str, Any]:
    try:
        response = requests.post(
            f"{BASE_URL}/ai/consultant/chat",
            json={"user_id": user_id, "message": message},
            timeout=300
        , proxies=PROXIES)

        if response.status_code == 200:
            return response.json()
        else:
            print(f"Ошибка сервера: {response.status_code} - {response.text}")
            # Заглушка для демонстрации, если бэкенд не отвечает
            return {
                "reply": "ИИ-консультант временно недоступен. Вот несколько общих советов:\n\n1. Обновите ваше резюме\n2. Изучите новые технологии в вашей области\n3. Посетите профессиональные мероприятия\n4. Создайте портфолио проектов",
                "courses": []
            }
    except requests.exceptions.RequestException as e:
        print(f"Ошибка подключения: {e}")
        return {
            "reply": "Не удается подключиться к серверу. Убедитесь, что бэкенд запущен.",
            "courses": []
        }


def add_microstep(user_id: int) -> bool:
    try:
        response = requests.post(
            f"{BASE_URL}/users/{user_id}/microstep",
            json={"done_on": None},
            timeout=5
        , proxies=PROXIES)
        return response.status_code == 200
    except requests.exceptions.RequestException as e:
        print(f"Ошибка соединения: {e}")
        return False
def get_achievements_catalog() -> Optional[Dict[str, Any]]:
    try:
        response = requests.get(f"{BASE_URL}/achievements/catalog", timeout=5, proxies=PROXIES)
        if response.status_code == 200:
            return response.json()
        print(f"Ошибка API (catalog): {response.status_code} - {response.text}")
    except requests.exceptions.RequestException as e:
        print(f"Ошибка соединения (catalog): {e}")
    return None
