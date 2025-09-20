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
warnings.filterwarnings("ignore", category=UserWarning, module="langchain_community.embeddings.fastembed") #он мне предупреждения выносил - решил вот так избежать :)
# сюда вот ключ нада
API_KEY = "sk-Kk1N_G-MJLcV2pEDgN2URg"  # <- вот сюда 
BASE_URL = "https://llm.t1v.scibox.tech/v1"

#вот тут json будет (сделаю потом загрузку через json)
SAMPLE_JSON = '''
[
  {
    "id": 1,
    "name": "Ноутбук Dell XPS 15",
    "category": "Электроника",
    "price_rub": 149990,
    "in_stock": true,
    "specs": {
      "cpu": "Intel Core i7-13700H",
      "ram_gb": 32,
      "storage_gb": 1024,
      "screen": "15.6\\" 4K OLED"
    },
    "description": "Мощный ультрабук для профессионалов: тонкий корпус, яркий дисплей, производительность на максимуме."
  },
  {
    "id": 2,
    "name": "Кофемашина DeLonghi Magnifica",
    "category": "Бытовая техника",
    "price_rub": 54990,
    "in_stock": false,
    "specs": {
      "type": "зерновая",
      "pressure_bar": 15,
      "tank_liters": 1.8,
      "programs": ["эспрессо", "американо", "капучино"]
    },
    "description": "Автоматическая кофемашина с функцией помола зерен и капучинатором."
  },
  {
    "id": 3,
    "name": "Фитнес-браслет Xiaomi Mi Band 8",
    "category": "Гаджеты",
    "price_rub": 3990,
    "in_stock": true,
    "specs": {
      "display": "AMOLED 1.62\\"",
      "battery_days": 14,
      "water_resistance": "5 ATM",
      "sensors": ["пульс", "сатурация", "шаги", "сон"]
    },
    "description": "Лёгкий и стильный браслет с длительным временем работы и точным мониторингом здоровья."
  }
]
'''

# Парсим json. Это для примера - потом убрать
try:
    data = json.loads(SAMPLE_JSON)
    print(f"Успешно загружено {len(data)} записей.")
except json.JSONDecodeError as e:
    print(f" Ошибка в JSON: {e}")
    exit(1)

# Преобразуем в Document
documents = []
for item in data:
    content = (
        f"Название: {item.get('name', 'не указано')}\n"
        f"Категория: {item.get('category', 'не указана')}\n"
        f"Цена: {item.get('price_rub', 'не указана')} руб.\n"
        f"В наличии: {'да' if item.get('in_stock', False) else 'нет'}\n"
        f"Описание: {item.get('description', 'нет описания')}\n"
    )
    specs = item.get("specs", {})
    if isinstance(specs, dict):
        content += "Характеристики:\n"
        for key, value in specs.items():
            if isinstance(value, list):
                value = ", ".join(map(str, value))
            content += f"  {key}: {value}\n"
    
    documents.append(Document(
        page_content=content.strip(),
        metadata={"id": item.get("id"), "name": item.get("name")}
    ))
#для чека 
print(f"Подготовлено {len(documents)} документов.")

#на чанки разбиваем аккуратнеько
text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
chunks = text_splitter.split_documents(documents)


#тут на эмбилдинги рабиваем
embeddings = FastEmbedEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    doc_embed_type="passage"
)

# Временная векторная база. Если надо будет - можно на пк закинуть потом 
vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

#LLM
llm = ChatOpenAI(
    model="Qwen2.5-72B-Instruct-AWQ",
    base_url=BASE_URL,
    api_key=API_KEY,
    temperature=0.7,
    max_tokens=512,
    top_p = 0.9
)

# Форматирование да ПРОМТ(потом поменять промт)
def format_docs(docs):
    return "\n\n".join(
        f"[ID {doc.metadata.get('id')}]: {doc.page_content.strip()}"
        for doc in docs
    )
#вот это мб поменять потом
template = """Ты — помощник по товарам. Ответь по контексту. Если не знаешь — скажи "Не указано".

Контекст:
{context}

Вопрос: {question}
Ответ:"""

prompt = ChatPromptTemplate.from_template(template)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

#геймифкация :)
print("\n" + "="*60)
print("Задавай вопросы по товарам. Введи 'выход', чтобы завершить.")
print("="*60)
#ну это вывод ответа и выход из цикла 
while True:
    question = input("\nВопрос: ").strip()
    if question.lower() in ["выход", "выйти", "exit", "quit", "q"]:
        print(" Пока!")
        break
    if not question:
        continue
    try:
        answer = rag_chain.invoke(question)
        print(f"\nОтвет:\n{answer}")
    except Exception as e:
        print(f"Ошибка: {e}")