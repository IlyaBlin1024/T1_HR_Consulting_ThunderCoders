import json
import os
import sys
from unittest.mock import mock_open, patch, MagicMock
from langchain_core.documents import Document
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from rag import load_data, create_documents

def test_load_data_valid():
    """Убедиться, что базовый сценарий — загрузка списка профилей — работает."""
    mock_json = '''
    [
        {"name": "Анна", "title": "Dev"},
        {"name": "Борис", "title": "ML"}
    ]
    '''
    with patch("builtins.open", mock_open(read_data=mock_json)):
        data = load_data("fake.json")
        assert len(data) == 2
        assert data[0]["name"] == "Анна"
    return True

def test_load_data_dict_input():
    """умеет обрабатывать JSON, в котором данные — словарь (объект), а не список."""
    mock_json = '''
    {
        "1": {"name": "Анна", "title": "Dev"},
        "2": {"name": "Борис", "title": "ML"}
    }
    '''
    with patch("builtins.open", mock_open(read_data=mock_json)):
        data = load_data("fake.json")
        assert len(data) == 2
        assert data[0]["name"] == "Анна"
    return True

def test_load_data_file_not_found():
    """Проверяет, что при попытке открыть несуществующий файл, функция выбрасывает FileNotFoundError"""
    with patch("builtins.open", side_effect=FileNotFoundError):
        try:
            load_data("nonexistent.json")
            return False  # Должно было упасть
        except FileNotFoundError:
            return True

def test_load_data_invalid_json():
    """Проверяет, что при битом JSON функция выбрасывает ValueError"""
    with patch("builtins.open", mock_open(read_data="invalid json {")):
        try:
            load_data("corrupt.json")
            return False
        except ValueError as e:
            assert "Ошибка в формате JSON" in str(e)
            return True


def test_create_documents_basic():
    """Проверяет, что функция create_documents корректно создаёт объект Document из минимального набора данных"""
    data = [
        {
            "id": "1",
            "name": "Анна",
            "title": "Python Developer",
            "location": "Москва",
            "skills": [],
            "stack": [],
            "desired_salary": None
        }
    ]
    docs = create_documents(data)
    assert len(docs) == 1
    doc = docs[0]
    assert "Имя: Анна" in doc.page_content
    assert doc.metadata["name"] == "Анна"
    assert doc.metadata["id"] == "1"
    return True

def test_create_documents_with_skills():
    """Проверяет, что функция корректно обрабатывает сложные поля: навыки, стек, зарплату — и включает их в page_content"""
    data = [
        {
            "id": "2",
            "name": "Борис",
            "skills": [
                {
                    "name": "Python",
                    "level": "Senior",
                    "years": 5,
                    "keywords": ["fastapi", "asyncio"]
                }
            ],
            "stack": ["Docker", "Kubernetes"],
            "desired_salary": 300000,
            "location": "СПб",
            "title": "DevOps"
        }
    ]
    docs = create_documents(data)
    content = docs[0].page_content
    assert "Навыки:" in content
    assert "Python (уровень: Senior, лет: 5)" in content
    assert "Ключевые слова: fastapi, asyncio" in content
    assert "Технологический стек: Docker, Kubernetes" in content
    assert "Ожидаемая зарплата: 300000" in content
    return True

def test_create_documents_empty_input():
    """Проверяет, что при пустом списке на входе функция возвращает пустой список документов — без ошибок"""
    docs = create_documents([])
    assert len(docs) == 0
    return True