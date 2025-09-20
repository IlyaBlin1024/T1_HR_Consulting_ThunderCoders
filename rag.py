import json
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="langchain_community.embeddings.fastembed")

#API ключ и URL
API_KEY = "sk-Kk1N_G-MJLcV2pEDgN2URg"
BASE_URL = "https://llm.t1v.scibox.tech/v1"

#Чтение данных
try:
    with open("profiles.json", "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    #Преобразуем из словаря в список, если нужно
    if isinstance(raw_data, dict):
        data = list(raw_data.values())
    else:
        data = raw_data
    print(f"Успешно загружено {len(data)} записей из profiles.json.")
except FileNotFoundError:
    print("Файл profiles.json не найден. Убедитесь, что он находится в текущей директории.")
    exit(1)
except json.JSONDecodeError as e:
    print(f"Ошибка в формате JSON: {e}")
    exit(1)

#Преобразуем в Document
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

    #Добавляем навыки
    skills = item.get("skills", [])
    if skills:
        content += "Навыки:\n"
        for skill in skills:
            keywords = ", ".join(skill.get("keywords", [])) if skill.get("keywords") else "—"
            content += f"  - {skill.get('name', 'неизвестно')} (уровень: {skill.get('level', '?')}, лет: {skill.get('years', '?')}) — Ключевые слова: {keywords}\n"

    #Добавляем стек
    stack = item.get("stack", [])
    if stack:
        content += f"Технологический стек: {', '.join(stack)}\n"

    #Добавляем зарплатные ожидания, если есть
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

print(f"Подготовлено {len(documents)} документов.")

#Разбиваем на чанки
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(documents)

#Создаём эмбеддинги
embeddings = FastEmbedEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    doc_embed_type="passage"
)

#Векторная база
vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

#LLM
llm = ChatOpenAI(
    model="Qwen2.5-72B-Instruct-AWQ",
    base_url=BASE_URL,
    api_key=API_KEY,
    temperature=0.7,
    max_tokens=512,
    top_p=0.9
)

#Форматирование документов для контекста
def format_docs(docs):
    return "\n\n".join(
        f"[ID {doc.metadata.get('id')}]: {doc.page_content.strip()}"
        for doc in docs
    )

#Промпт
template = """Ты — HR-ассистент, который помогает подбирать сотрудников по их профилям. Ответь по контексту. Если информации нет — скажи "Не указано".

Контекст:
{context}

Вопрос: {question}
Ответ:"""

prompt = ChatPromptTemplate.from_template(template)

#RAG цепочка
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

#Интерактивный цикл
print("\n" + "="*60)
print("Задавай вопросы по профилям сотрудников. Введи 'выход', чтобы завершить.")
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