import json
import os 
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
import warnings

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

#Полный путь к profiles.json
PROFILE_PATH = os.path.join(SCRIPT_DIR, "profiles.json")


warnings.filterwarnings("ignore", category=UserWarning, module="langchain_community.embeddings.fastembed")

API_KEY = "sk-Kk1N_G-MJLcV2pEDgN2URg"
BASE_URL = "https://llm.t1v.scibox.tech/v1"

def load_data(filepath=PROFILE_PATH):
    """Загружает данные из JSON-файла"""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
        if isinstance(raw_data, dict):
            return list(raw_data.values())
        return raw_data
    except FileNotFoundError:
        raise FileNotFoundError(f"Файл {filepath} не найден.")
    except json.JSONDecodeError as e:
        raise ValueError(f"Ошибка в формате JSON: {e}")

def create_documents(data):
    """Преобразует сырые данные в список Document"""
    documents = []
    for item in data:
        content = (
            f"Имя: {item.get('name', 'не указано')}\n"
            f"Должность: {item.get('title', 'не указана')}\n"
            f"Локация: {item.get('location', 'не указана')}\n"
            f"Опыт работы (лет): {item.get('years_experience', 'не указан')}\n"
            f"Готовность к ротации: {item.get('open_to_rotation', 'неизвестно')}\n"
            f"Оценка готовности: {item.get('readiness_score', 'не указана')}\n"
            f"Краткое резюме: {item.get('summary', 'нет описания')}\n"
        )

        skills = item.get("skills", [])
        if skills:
            content += "Навыки:\n"
            for skill in skills:
                keywords = ", ".join(skill.get("keywords", [])) if skill.get("keywords") else "—"
                content += f"  - {skill.get('name', 'неизвестно')} (уровень: {skill.get('level', '?')}, лет: {skill.get('years', '?')}) — Ключевые слова: {keywords}\n"

        stack = item.get("stack", [])
        if stack:
            content += f"Технологический стек: {', '.join(stack)}\n"

        salary = item.get("desired_salary")
        if salary:
            content += f"Ожидаемая зарплата: {salary}\n"

        documents.append(Document(
            page_content=content.strip(),
            metadata={
                "id": item.get("id", "unknown"),
                "name": item.get("name", "unknown"),
                "title": item.get("title", "unknown"),
                "location": item.get("location", "unknown")
            }
        ))
    return documents

def create_rag_chain(documents, model_name="Qwen2.5-72B-Instruct-AWQ"):
    """Создаёт и возвращает RAG-цепочку"""

    chunks = documents

    embeddings = FastEmbedEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        doc_embed_type="passage"
    )

    vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    llm = ChatOpenAI(
        model=model_name,
        base_url=BASE_URL,
        api_key=API_KEY,
        temperature=0.7,
        max_tokens=512,
        top_p=0.9
    )

    def format_docs(docs):
        return "\n\n".join(
            f"[ID {doc.metadata.get('id')}]: {doc.page_content.strip()}"
            for doc in docs
        )

    template = """Ты — HR-ассистент... {context} ... {question}"""
    prompt = ChatPromptTemplate.from_template(template)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain

#Главный цикл ТОЛЬКО при запуске файла
if __name__ == "__main__":
    print("Загрузка данных...")
    data = load_data()
    print(f"Загружено {len(data)} записей.")

    documents = create_documents(data)
    print(f"Создано {len(documents)} документов.")

    rag_chain = create_rag_chain(documents)

    print("\n" + "="*60)
    print("Задавай вопросы... Введи 'выход', чтобы завершить.")
    print("="*60)

    while True:
        question = input("\nВопрос: ").strip()
        if question.lower() in ["выход", "выйти", "exit", "quit", "q"]:
            print("Пока!")
            break
        if not question:
            continue
        try:
            answer = rag_chain.invoke(question)
            print(f"\nОтвет:\n{answer}")
        except Exception as e:
            print(f"Ошибка: {e}")