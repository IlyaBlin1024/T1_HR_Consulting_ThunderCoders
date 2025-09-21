import os
from dotenv import load_dotenv

# Загружаем переменные окружения из .env
load_dotenv()

SCIBOX_API_KEY = os.getenv("SCIBOX_API_KEY")

if not SCIBOX_API_KEY:
    raise RuntimeError("Set SCIBOX_API_KEY in .env or environment")