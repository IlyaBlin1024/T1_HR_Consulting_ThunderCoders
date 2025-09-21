# ============================== ИМПОРТЫ БИБЛИОТЕК ==============================
from fastapi import FastAPI, HTTPException, Depends, Body  # веб-фреймворк и утилиты для ошибок/зависимостей/тел
from fastapi.middleware.cors import CORSMiddleware          # middleware для CORS, чтобы фронт (в т.ч. Gradio) звал API
from pydantic import BaseModel, EmailStr, Field             # модели валидации входа/выхода и тип для email
from typing import List, Optional, Dict, Any, Generator, TypedDict  # типы для аннотаций, Generator для dependency, TypedDict для стейта
from sqlalchemy import (                                     # ядро SQLAlchemy (DDL/DML)
    create_engine, Column, Integer, String, Date, DateTime,
    Float, ForeignKey, UniqueConstraint, Text
)
from sqlalchemy.orm import sessionmaker, declarative_base, relationship, Session, Mapped, mapped_column  # ORM: фабрика сессий, базовый класс, relationship
from datetime import datetime, date, timedelta                # работа с датой/временем
from dotenv import load_dotenv                               # загрузка .env параметров
import os                                                    # доступ к переменным окружения/файлам
from openai import OpenAI                                    # клиент OpenAI-совместимого API (Scibox)
from langgraph.graph import StateGraph, END                  # LangGraph: построение графа состояний для ИИ-консультанта



# ============================== КОНФИГ И ОКРУЖЕНИЕ =============================
load_dotenv()                                                # подгружаем переменные окружения из .env если есть
SCIBOX_API_KEY = os.getenv("SCIBOX_API_KEY", "").strip()
SCIBOX_BASE_URL = os.getenv("SCIBOX_BASE_URL", "http://176.119.5.23:4000/v1")  # URL Scibox по умолчанию

print(f"SCIBOX_API_KEY: {'установлен' if SCIBOX_API_KEY else 'не установлен'}")
print(f"SCIBOX_BASE_URL: {SCIBOX_BASE_URL}")
# ============================== ИНИЦИАЛИЗАЦИЯ ПРИЛОЖЕНИЯ =======================
app = FastAPI(title="Career Backend", version="1.2.2")       # создаём экземпляр FastAPI с метаданными

app.add_middleware(
    # подключаем CORS-мидлварь
    CORSMiddleware,                                          # сам класс мидлвари
    allow_origins=["*"],                                     # разрешаем любые источники (можно сузить на проде)
    allow_credentials=True,                                  # разрешаем передавать креденшлы
    allow_methods=["*"],                                     # разрешаем любые HTTP-методы
    allow_headers=["*"],                                     # разрешаем любые заголовки
)

# ============================== НАСТРОЙКА БАЗЫ ДАННЫХ ==========================
DB_PATH = os.path.join("storage", "app.db")                  # путь к файлу SQLite в каталоге storage
os.makedirs("storage", exist_ok=True)                        # создаём каталог storage, если его нет
engine = create_engine(f"sqlite:///{DB_PATH}", echo=False, future=True)  # создаём SQLAlchemy-движок SQLite
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)  # фабрика сессий (ручной commit)
Base = declarative_base()                                     # базовый класс для ORM-моделей

# ============================== ORM-МОДЕЛИ ДАННЫХ (Typed ORM 2.0) ==============================
class User(Base):
    __tablename__ = "users"                                  # имя таблицы в БД

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)                 # числовой первичный ключ + индекс
    email: Mapped[str] = mapped_column(String, unique=True, nullable=False)                # e-mail пользователя, уникальный и обязательный
    phone: Mapped[Optional[str]] = mapped_column(String, nullable=True)                    # телефон (может отсутствовать)
    full_name: Mapped[str] = mapped_column(String, nullable=False)                         # ФИО, обязательное поле
    department: Mapped[Optional[str]] = mapped_column(String, nullable=True)               # подразделение (произвольная строка)
    position: Mapped[Optional[str]] = mapped_column(String, nullable=True)                 # должность (произвольная строка)
    grade: Mapped[Optional[str]] = mapped_column(String, nullable=True)                    # грейд/уровень
    current_roles: Mapped[Optional[str]] = mapped_column(Text, nullable=True)              # текущие роли/обязанности, большой текст
    experience_years: Mapped[float] = mapped_column(Float, default=0.0)                    # стаж в годах, по умолчанию 0.0
    resume_text: Mapped[Optional[str]] = mapped_column(Text, nullable=True)                # текст резюме, может быть пустым
    profile_photo_url: Mapped[Optional[str]] = mapped_column(String, nullable=True)        # ссылка на фото профиля
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)        # дата/время создания записи (UTC)
    updated_at: Mapped[datetime] = mapped_column(                                          # колонка последнего обновления
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )

    # связи (Typed) — коллекции дочерних сущностей. cascade удаляет их вместе с пользователем
    skills: Mapped[list["Skill"]] = relationship(back_populates="user", cascade="all, delete-orphan")             # 1:N навыки
    projects: Mapped[list["Project"]] = relationship(back_populates="user", cascade="all, delete-orphan")         # 1:N проекты
    endorsements: Mapped[list["Endorsement"]] = relationship(back_populates="user", cascade="all, delete-orphan") # 1:N подтверждения навыков
    certificates: Mapped[list["Certificate"]] = relationship(back_populates="user", cascade="all, delete-orphan") # 1:N сертификаты
    achievements: Mapped[list["UserAchievement"]] = relationship(back_populates="user", cascade="all, delete-orphan")  # 1:N ачивки
    microsteps: Mapped[list["Microstep"]] = relationship(back_populates="user", cascade="all, delete-orphan")     # 1:N микрошаги (для стриков)
    chat_messages: Mapped[list["ChatMessage"]] = relationship(back_populates="user", cascade="all, delete-orphan") # 1:N история чата ИИ

class Skill(Base):
    __tablename__ = "skills"                                 # имя таблицы

    id: Mapped[int] = mapped_column(Integer, primary_key=True)                              # PK навыка
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=False)            # внешний ключ на users.id
    name: Mapped[str] = mapped_column(String, nullable=False, index=True)                   # название/метка навыка; индекс для быстрых выборок
    level: Mapped[Optional[str]] = mapped_column(String, nullable=True)                     # уровень владения (например, Junior/Middle/Senior)

    user: Mapped["User"] = relationship(back_populates="skills")                             # обратная связь к владельцу навыка

class Project(Base):
    __tablename__ = "projects"                              # имя таблицы

    id: Mapped[int] = mapped_column(Integer, primary_key=True)                              # PK проекта
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=False)            # FK на пользователя
    title: Mapped[str] = mapped_column(String, nullable=False)                              # название проекта (обязательное)
    role: Mapped[Optional[str]] = mapped_column(String, nullable=True)                      # роль в проекте (опционально)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)                 # описание/детали проекта
    result_kpi: Mapped[Optional[str]] = mapped_column(String, nullable=True)                # достигнутый KPI/результат для ачивок

    user: Mapped["User"] = relationship(back_populates="projects")                          # обратная связь к пользователю

class Endorsement(Base):
    __tablename__ = "endorsements"                          # имя таблицы

    id: Mapped[int] = mapped_column(Integer, primary_key=True)                              # PK подтверждения
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=False)            # FK на владельца
    skill_name: Mapped[str] = mapped_column(String, nullable=False)                         # какой навык подтверждён
    from_team: Mapped[Optional[str]] = mapped_column(String, nullable=True)                 # от какой команды/подразделения
    endorsed_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)        # когда подтверждено (UTC)

    user: Mapped["User"] = relationship(back_populates="endorsements")                      # обратная связь к пользователю

class Certificate(Base):
    __tablename__ = "certificates"                          # имя таблицы

    id: Mapped[int] = mapped_column(Integer, primary_key=True)                              # PK сертификата
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=False)            # FK на пользователя
    name: Mapped[str] = mapped_column(String, nullable=False)                               # название сертификата
    issued_by: Mapped[Optional[str]] = mapped_column(String, nullable=True)                 # кем выдан (организация)
    valid_until: Mapped[Optional[date]] = mapped_column(Date, nullable=True)                # срок действия (может отсутствовать)

    user: Mapped["User"] = relationship(back_populates="certificates")                      # обратная связь к пользователю

class UserAchievement(Base):
    __tablename__ = "user_achievements"                    # имя таблицы

    id: Mapped[int] = mapped_column(Integer, primary_key=True)                              # PK ачивки
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=False)            # FK на пользователя
    code: Mapped[str] = mapped_column(String, nullable=False)                               # код ачивки (например, 'profile_master')
    level: Mapped[str] = mapped_column(String, nullable=False)                              # уровень ('бронза'/'серебро'/...)
    xp: Mapped[int] = mapped_column(Integer, nullable=False)                                # количество очков опыта, начисленных за ачивку
    obtained_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)  # когда выдана (с индексом для сортировки)

    user: Mapped["User"] = relationship(back_populates="achievements")                      # обратная связь к пользователю

    __table_args__ = (UniqueConstraint("user_id", "code", "level", name="uq_user_ach_level"),)  # уникальность уровня ачивки на пользователя

class Microstep(Base):
    __tablename__ = "microsteps"                            # имя таблицы

    id: Mapped[int] = mapped_column(Integer, primary_key=True)                              # PK микрошагa
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=False)            # FK на пользователя
    done_on: Mapped[date] = mapped_column(Date, nullable=False, index=True)                 # дата выполнения (1 шаг в день/неделю)

    user: Mapped["User"] = relationship(back_populates="microsteps")                        # обратная связь к пользователю

    __table_args__ = (UniqueConstraint("user_id", "done_on", name="uq_user_day"),)          # запрет дублировать дату для одного пользователя

class ChatMessage(Base):
    __tablename__ = "chat_messages"                         # имя таблицы

    id: Mapped[int] = mapped_column(Integer, primary_key=True)                              # PK сообщения
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=False)            # FK на пользователя
    role: Mapped[str] = mapped_column(String, nullable=False)                               # роль в чате ('user' или 'assistant')
    content: Mapped[str] = mapped_column(Text, nullable=False)                              # текстовое содержимое сообщения
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)  # время создания (UTC) + индекс

    user: Mapped["User"] = relationship(back_populates="chat_messages")                     # обратная связь к пользователю


# ============================== СОЗДАЁМ ТАБЛИЦЫ ================================
Base.metadata.create_all(bind=engine)                         # создаём физические таблицы в SQLite, если их нет

# ============================== ФУНКЦИЯ ВЫДАЧИ СЕССИИ ==========================
def get_db() -> Generator[Session, None, None]:              # зависимость FastAPI: генератор сессии БД
    db = SessionLocal()                                      # открываем новую сессию
    try:                                                     # защищаемся от исключений
        yield db                                             # отдаём сессию обработчику
    finally:                                                 # по завершении запроса
        db.close()                                           # закрываем сессию

# ============================== КАТАЛОГ АЧИВОК ================================
ACHIEVEMENTS_CATALOG: Dict[str, Dict[str, Any]] = {          # словарь правил ачивок и XP
    "profile_master": {                                      # ачивка «Мастер профиля»
        "title": "Мастер профиля",                           # заголовок
        "levels": [("бронза", 30, 0.40), ("серебро", 60, 0.60), ("золото", 90, 0.80), ("алмаз", 140, 0.95), ("платина", 200, 1.00)],  # уровни/XP/пороги
        "absolute": ("абсолютный", 320)                      # абсолютный уровень и его XP
    },
    "skill_map": { "title": "Навыковая карта", "thresholds": [(5, 40), (10, 80), (15, 120), (20, 180), (25, 260)] },  # пороги по числу навыков
    "endorsed_skills": { "title": "Подтверждённые навыки", "thresholds": [(3, 40), (6, 80), (10, 120), (15, 180), (20, 260)] },  # пороги по эндорсментам
    "certified": { "title": "Сертифицированный специалист", "thresholds": [(1, 40), (2, 80), (3, 120), (5, 180), (7, 260)], "absolute": ("абсолютный", 420) },  # сертификаты
    "project_impact": { "title": "Проект с результатом", "thresholds": [(1, 60), (2, 120), (3, 180), (4, 260), (5, 360)], "absolute": ("абсолютный", 600) },     # проекты с KPI
    "project_portfolio": { "title": "Портфель проектов", "thresholds": [(1, 40), (2, 80), (3, 120), (5, 180), (8, 260)] },                                       # просто количество проектов
    "soft_endorse": { "title": "Софт-скиллы подтверждены", "thresholds": [(2, 30), (4, 60), (6, 90), (8, 140), (10, 200)] },                                     # подтверждения soft-навыков
    "language_readiness": { "title": "Языковая готовность", "levels": [("A2", 20), ("B1", 40), ("B2", 80), ("C1", 140), ("C2", 200)] },                         # язык и XP
    "availability": { "title": "Доступен для проектов", "thresholds": [(1, 20), (2, 40), (3, 80), (6, 140), (12, 200)] },                                       # месяцы доступности
    "mentor": { "title": "Наставник", "thresholds": [(3, 60), (6, 120), (10, 180), (15, 260), (20, 360)], "absolute": ("абсолютный", 600) },                    # менторство
    "compliance": { "title": "Комплаенс без сюрпризов", "thresholds": [(1, 30), (2, 60), (3, 90), (4, 140), (5, 200)], "absolute": ("абсолютный", 320) }        # комплаенс
}

# ============================== ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ========================
def mandatory_profile_fields_filled(u: User) -> float:       # доля заполненности ключевых полей профиля
    required = [u.email, u.full_name]                        # минимально обязательные поля
    optional_required = [u.position, u.department, u.grade]  # квази-обязательные поля из анкеты
    total = len(required) + len(optional_required)           # общее количество учитываемых полей
    filled = sum(1 for x in required + optional_required if x and str(x).strip())  # сколько реально заполнено
    return filled / total if total else 0.0                  # возвращаем долю (0..1), защита от деления на ноль

def language_level_to_xp(level: str) -> int:                 # конвертируем уровень языка в XP
    mapping = {"A2": 20, "B1": 40, "B2": 80, "C1": 140, "C2": 200}  # соответствие уровней
    return mapping.get(level.upper(), 0)                     # неизвестный уровень даёт 0 XP

def compute_weekly_streak(steps: List[date]) -> Dict[str, Any]:  # считаем недельный «стрик» по датам микрошагов
    if not steps:                                            # если шагов нет
        return {"completed_weeks": 0, "master_checkpoints": 0, "status": "пауза"}  # пустой результат
    steps_sorted = sorted(set(steps))                        # сортируем и убираем дубликаты
    start_week = steps_sorted[0] - timedelta(days=steps_sorted[0].weekday())  # понедельник первой недели
    last_week = steps_sorted[-1] - timedelta(days=steps_sorted[-1].weekday()) # понедельник последней недели
    weeks, cur = [], start_week                              # инициализируем список недель и указатель
    while cur <= last_week:                                  # идём от первой недели к последней
        weeks.append(cur)                                    # добавляем начало недели
        cur += timedelta(weeks=1)                            # переходим к следующей неделе
    completed_weeks = sum(                                   # считаем недели, где был хотя бы один микрошаг
        1 for w in weeks if any(w <= d <= w + timedelta(days=6) for d in steps_sorted)
    )
    master_checkpoints = completed_weeks // 4                # каждый 4-й завершённый блок недель — чекпоинт
    status = "активен" if (steps_sorted[-1] >= (date.today() - timedelta(days=7))) else "пауза"  # активность по последней неделе
    return {"completed_weeks": completed_weeks, "master_checkpoints": master_checkpoints, "status": status}  # отдаём метрики

def xp_from_streak(streak: Dict[str, Any]) -> int:           # перевод метрик стрика в XP
    return streak["completed_weeks"] * 10 + streak["master_checkpoints"] * 50  # 10 XP за неделю + 50 XP за чекпоинт


def scibox_client() -> Optional[OpenAI]:
    """Создаём клиента Scibox (если есть ключ) с улучшенным логированием"""
    if not SCIBOX_API_KEY:
        print("Scibox API ключ не найден. Установите переменную окружения SCIBOX_API_KEY")
        return None

    if not SCIBOX_BASE_URL:
        print("Scibox base URL не найден. Установите переменную окружения SCIBOX_BASE_URL")
        return None

    try:
        print(f"Попытка подключения к Scibox: {SCIBOX_BASE_URL}")
        client = OpenAI(api_key=SCIBOX_API_KEY, base_url=SCIBOX_BASE_URL)

        # Упрощенная проверка доступности API (без запроса списка моделей)
        # Просто создаем клиента - если параметры неверные, это вызовет исключение
        print("Клиент Scibox успешно создан")
        return client
    except Exception as e:
        print(f"Ошибка при создании Scibox клиента: {type(e).__name__}: {str(e)}")
        return None

# ============================== Pydantic-СХЕМЫ (CRUD) ==========================
class SkillIn(BaseModel):                                    # входная схема «Навык»
    name: str = Field(..., description="Название навыка")    # название навыка (обязательно)
    level: Optional[str] = Field(None, description="Уровень владения")  # уровень (опционально)

class ProjectIn(BaseModel):                                  # входная схема «Проект»
    title: str                                               # название проекта (обязательно)
    role: Optional[str] = None                               # роль
    description: Optional[str] = None                        # описание
    result_kpi: Optional[str] = None                         # результат/KPI

class CertificateIn(BaseModel):                              # входная схема «Сертификат»
    name: str                                                # название (обязательно)
    issued_by: Optional[str] = None                          # кем выдан
    valid_until: Optional[date] = None                       # срок действия

class UserCreate(BaseModel):                                 # входная схема «Создать пользователя»
    email: EmailStr                                          # почта (обязательно)
    full_name: str                                           # ФИО (обязательно)
    phone: Optional[str] = None                              # телефон
    department: Optional[str] = None                         # подразделение
    position: Optional[str] = None                           # должность
    grade: Optional[str] = None                              # грейд
    experience_years: Optional[float] = 0.0                  # стаж (по умолчанию 0.0)
    resume_text: Optional[str] = None                        # резюме
    profile_photo_url: Optional[str] = None                  # фото
    skills: List[SkillIn] = []                               # список навыков (по умолчанию пуст)
    projects: List[ProjectIn] = []                           # список проектов (по умолчанию пуст)
    certificates: List[CertificateIn] = []                   # список сертификатов (по умолчанию пуст)

class UserPatch(BaseModel):                                  # входная схема «Обновить пользователя»
    email: Optional[EmailStr] = None                         # почта (опционально)
    full_name: Optional[str] = None                          # ФИО (опционально)
    phone: Optional[str] = None                              # телефон
    department: Optional[str] = None                         # подразделение
    position: Optional[str] = None                           # должность
    grade: Optional[str] = None                              # грейд
    experience_years: Optional[float] = None                 # стаж
    resume_text: Optional[str] = None                        # резюме
    profile_photo_url: Optional[str] = None                  # фото
    skills: Optional[List[SkillIn]] = None                   # навыки
    projects: Optional[List[ProjectIn]] = None               # проекты
    certificates: Optional[List[CertificateIn]] = None       # сертификаты

class UserPublic(BaseModel):                                 # публичная схема «Пользователь»
    id: int                                                  # идентификатор
    email: EmailStr                                          # почта
    full_name: str                                           # ФИО
    phone: Optional[str]                                     # телефон
    department: Optional[str]                                # подразделение
    position: Optional[str]                                  # должность
    grade: Optional[str]                                     # грейд
    experience_years: float                                  # стаж
    resume_text: Optional[str]                               # резюме
    profile_photo_url: Optional[str]                         # фото
    class Config:                                            # конфигурация pydantic-модели
        from_attributes = True                               # разрешаем строить из ORM-объектов напрямую

class AchievementPublic(BaseModel):                          # схема «Ачивка для фронта»
    code: str                                                # код
    title: str                                               # заголовок
    level: str                                               # уровень
    xp: int                                                  # очки XP
    obtained_at: datetime                                    # дата получения

class DashboardResponse(BaseModel):                          # схема ответа «Личный кабинет»
    user: UserPublic                                         # данные пользователя
    progress_percent: float                                  # процент заполнения профиля
    total_xp: int                                            # суммарный XP (включая стрик)
    achievements: List[AchievementPublic]                    # список ачивок пользователя
    recommended_achievements: List[str]                      # рекомендации по закрытию следующих ачивок
    llm_tips: Optional[str]                                  # советы ИИ (если был вызов Scibox)

# ============================== CRUD-ХЕЛПЕРЫ ДЛЯ СВЯЗАННЫХ ТАБЛИЦ =============
def upsert_skills(db: Session, user: User, skills_in: List[SkillIn]) -> None:  # перезапись набора навыков
    user.skills.clear()                                   # удаляем текущие навыки пользователя
    for s in skills_in:                                   # пробегаем входные навыки
        user.skills.append(Skill(name=s.name.strip(), level=(s.level or "").strip()))  # добавляем ORM-объект Skill

def upsert_projects(db: Session, user: User, projects_in: List[ProjectIn]) -> None:  # перезапись проектов
    user.projects.clear()                                 # удаляем текущие проекты
    for p in projects_in:                                 # пробегаем входные проекты
        user.projects.append(Project(                     # добавляем ORM-объект Project
            title=p.title.strip(),                        # название
            role=(p.role or "").strip(),                  # роль (безопасная строка)
            description=(p.description or "").strip(),    # описание (безопасная строка)
            result_kpi=(p.result_kpi or "").strip()       # KPI/итог (безопасная строка)
        ))

def upsert_certificates(db: Session, user: User, certs_in: List[CertificateIn]) -> None:  # перезапись сертификатов
    user.certificates.clear()                            # удаляем текущие сертификаты
    for c in certs_in:                                   # пробегаем входные сертификаты
        user.certificates.append(Certificate(            # добавляем ORM-объект Certificate
            name=c.name.strip(),                         # название
            issued_by=(c.issued_by or "").strip(),       # организация-выдаватель
            valid_until=c.valid_until                    # срок действия (дата или None)
        ))

# ============================== ВЫДАЧА АЧИВОК И ПРОГРЕСС ======================
def _issue(db: Session, user: User, code: str, level: str, xp: int) -> None:   # вспомогательная выдача ачивки
    exists = db.query(UserAchievement).filter_by(user_id=user.id, code=code, level=level).first()  # ищем дубликат
    if exists:                                              # если уже есть
        return                                              # ничего не делаем
    db.add(UserAchievement(user_id=user.id, code=code, level=level, xp=xp))  # создаём новую запись ачивки

def calculate_and_issue_achievements(db: Session, user: User) -> int:          # основной расчёт ачивок и XP
    fill = mandatory_profile_fields_filled(user)            # доля заполненности ключевых полей
    for level_name, xp, threshold in ACHIEVEMENTS_CATALOG["profile_master"]["levels"]:  # обходим уровни «Мастер профиля»
        ok = fill >= threshold                              # выполняется ли порог по заполненности
        if level_name == "платина":                         # для платины требуем фото и резюме
            ok = ok and (bool(user.profile_photo_url) and bool(user.resume_text))  # оба поля должны быть заданы
        if ok:                                              # если условие выполнено
            _issue(db, user, "profile_master", level_name, xp)  # выдаём ачивку соответствующего уровня

    sc = len(user.skills)                                   # количество навыков
    for t, xp in ACHIEVEMENTS_CATALOG["skill_map"]["thresholds"]:  # пороги «Навыковой карты»
        if sc >= t:                                         # если выполняем порог
            _issue(db, user, "skill_map", f"{t}+", xp)      # выдаём уровень по порогу

    ec = len(user.endorsements)                             # количество эндорсментов
    for t, xp in ACHIEVEMENTS_CATALOG["endorsed_skills"]["thresholds"]:  # пороги «Подтверждённых навыков»
        if ec >= t:                                         # проверяем порог
            _issue(db, user, "endorsed_skills", f"{t}+", xp)  # выдаём ачивку

    cc = len(user.certificates)                             # количество сертификатов
    for t, xp in ACHIEVEMENTS_CATALOG["certified"]["thresholds"]:  # пороги «Сертифицированного специалиста»
        if cc >= t:                                         # проверяем порог
            _issue(db, user, "certified", f"{t}+", xp)      # выдаём ачивку

    proj_with_kpi = sum(1 for p in user.projects if p.result_kpi)  # проекты с заполненным KPI
    for t, xp in ACHIEVEMENTS_CATALOG["project_impact"]["thresholds"]:  # пороги «Проект с результатом»
        if proj_with_kpi >= t:                               # проверяем порог
            _issue(db, user, "project_impact", f"{t}+", xp)  # выдаём ачивку

    proj_total = len(user.projects)                          # общее число проектов
    for t, xp in ACHIEVEMENTS_CATALOG["project_portfolio"]["thresholds"]:  # пороги «Портфеля проектов»
        if proj_total >= t:                                  # выполняем порог
            _issue(db, user, "project_portfolio", f"{t}+", xp)  # выдаём ачивку

    soft_count = sum(1 for e in user.endorsements if e.skill_name.lower().startswith("soft:"))  # soft-эндорсменты
    for t, xp in ACHIEVEMENTS_CATALOG["soft_endorse"]["thresholds"]:  # пороги «Софт-скиллы подтверждены»
        if soft_count >= t:                                   # проверка порога
            _issue(db, user, "soft_endorse", f"{t}+", xp)     # выдача

    for sk in user.skills:                                   # просмотр навыков
        if sk.name.lower().startswith("lang:"):              # языковой маркер "lang:XX=Level"
            lvl = sk.name.split(":", 1)[1].split("=")[-1].upper()  # извлекаем уровень справа от "="
            x = language_level_to_xp(lvl)                    # переводим уровень в XP
            if x > 0:                                        # если уровень распознан
                _issue(db, user, "language_readiness", lvl, x)  # выдаём ачивку языка

    avail_months = sum(1 for sk in user.skills if sk.name.lower().startswith("availability:"))  # месяцы доступности
    for t, xp in ACHIEVEMENTS_CATALOG["availability"]["thresholds"]:  # пороги «Доступен для проектов»
        if avail_months >= t:                               # проверка порога
            _issue(db, user, "availability", f"{t}m+", xp) # выдача

    mentor_sessions = 0                                     # счётчик менторских сессий
    for sk in user.skills:                                  # проходим навыки
        if sk.name.lower().startswith("mentor_sessions:"):  # формат "mentor_sessions:N"
            try:                                            # пытаемся распарсить N
                mentor_sessions += int(sk.name.split(":")[1])  # суммируем значение
            except Exception:                                # если формат кривой
                pass                                        # просто пропускаем
    for t, xp in ACHIEVEMENTS_CATALOG["mentor"]["thresholds"]:  # пороги «Наставник»
        if mentor_sessions >= t:                             # проверяем порог
            _issue(db, user, "mentor", f"{t}+", xp)          # выдаём уровень

    compliance_steps = sum(1 for sk in user.skills if sk.name.lower().startswith("compliance:step"))  # шаги комплаенса
    for t, xp in ACHIEVEMENTS_CATALOG["compliance"]["thresholds"]:  # пороги «Комплаенс без сюрпризов»
        if compliance_steps >= t:                             # проверяем порог
            _issue(db, user, "compliance", f"step{t}", xp)    # выдаём ачивку

    total_xp = sum(a.xp for a in user.achievements)          # суммируем XP из всех выданных ачивок
    streak = compute_weekly_streak([m.done_on for m in user.microsteps])  # считаем метрики стрика
    total_xp += xp_from_streak(streak)                        # добавляем XP за стрик
    return total_xp                                           # возвращаем общий XP

def profile_progress_percent(u: User) -> float:              # функция процента заполнения профиля
    fill = mandatory_profile_fields_filled(u)                # берём долю обязательных полей
    bonus = (0.05 if u.profile_photo_url else 0.0) + (0.10 if u.resume_text else 0.0)  # бонусы за фото/резюме
    return round(min(100.0, (fill * 100.0) + (bonus * 100.0)), 2)  # итоговый процент с ограничением 100% и округлением

def recommend_achievements(u: User) -> List[str]:            # простые рекомендации по ачивкам
    recs: List[str] = []                                     # инициализируем список
    if mandatory_profile_fields_filled(u) < 0.8:             # если заполненность <80%
        recs.append("profile_master")                        # совет закрыть «Мастер профиля»
    if len(u.skills) < 10:                                   # мало навыков
        recs.append("skill_map")                             # совет расширить «Навыковую карту»
    if not any(p.result_kpi for p in u.projects):            # нет KPI в проектах
        recs.append("project_impact")                        # совет оформить результаты
    if len(u.certificates) == 0:                             # нет сертификатов
        recs.append("certified")                             # совет получить сертификат
    return recs                                              # возвращаем список кодов

# ============================== CRUD ENDPOINTS ДЛЯ ПОЛЬЗОВАТЕЛЕЙ ===============
@app.post("/users", response_model=UserPublic)               # создание пользователя
def create_user(payload: UserCreate, db: Session = Depends(get_db)):  # зависимость на сессию БД
    if db.query(User).filter_by(email=str(payload.email)).first():  # проверяем уникальность email
        raise HTTPException(status_code=409, detail="User with this email exists")  # конфликт если уже есть
    user = User(                                              # создаём ORM-объект пользователя
        email=str(payload.email), full_name=payload.full_name, phone=payload.phone,  # маппим поля
        department=payload.department, position=payload.position, grade=payload.grade,  # орг-поля
        experience_years=payload.experience_years or 0.0,     # стаж (защищённый от None)
        resume_text=(payload.resume_text or ""),               # резюме (пустая строка, если None)
        profile_photo_url=(payload.profile_photo_url or "")    # фото (пустая строка, если None)
    )
    db.add(user)                                              # добавляем в сессию
    db.flush()                                                # синхронно получаем id без коммита
    upsert_skills(db, user, payload.skills)                   # сохраняем навыки
    upsert_projects(db, user, payload.projects)               # сохраняем проекты
    upsert_certificates(db, user, payload.certificates)       # сохраняем сертификаты
    db.commit()                                               # фиксируем транзакцию
    db.refresh(user)                                          # обновляем объект из БД
    return user                                               # отдаём публичную модель

@app.get("/users/{user_id}", response_model=UserPublic)      # получить пользователя по id
def get_user(user_id: int, db: Session = Depends(get_db)):   # зависимость на сессию БД
    user = db.get(User, user_id)                             # ищем по первичному ключу
    if not user:                                             # если не найден
        raise HTTPException(status_code=404, detail="User not found")  # бросаем 404
    return user                                              # возвращаем пользователя

@app.put("/users/{user_id}", response_model=UserPublic)      # обновить пользователя
def update_user(user_id: int, payload: UserPatch, db: Session = Depends(get_db)):  # зависимость на БД
    user = db.get(User, user_id)                             # ищем пользователя
    if not user:                                             # если нет такого
        raise HTTPException(status_code=404, detail="User not found")  # 404
    if payload.email and payload.email != user.email:        # если меняем email
        if db.query(User).filter_by(email=str(payload.email)).first():  # проверяем уникальность нового email
            raise HTTPException(status_code=409, detail="User with this email exists")  # конфликт
        user.email = str(payload.email)                      # применяем новый email
    for attr in ["full_name", "phone", "department", "position", "grade", "experience_years", "resume_text", "profile_photo_url"]:  # перечисляем обновляемые поля
        val = getattr(payload, attr)                         # достаём значение из payload
        if val is not None:                                  # если значение передано
            setattr(user, attr, val)                         # присваиваем пользователю
    if payload.skills is not None:                           # если передан список навыков
        upsert_skills(db, user, payload.skills)              # перезаписываем навыки
    if payload.projects is not None:                         # если передан список проектов
        upsert_projects(db, user, payload.projects)          # перезаписываем проекты
    if payload.certificates is not None:                     # если передан список сертификатов
        upsert_certificates(db, user, payload.certificates)  # перезаписываем сертификаты
    db.commit()                                              # сохраняем изменения
    db.refresh(user)                                         # обновляем объект
    return user                                              # отдаём пользователя

@app.post("/users/{user_id}/endorse", response_model=dict)   # добавить эндорсмент навыка
def endorse_skill(user_id: int, skill_name: str = Body(..., embed=True), from_team: str = Body("", embed=True), db: Session = Depends(get_db)):  # читаем тело запроса
    user = db.get(User, user_id)                             # проверяем, что пользователь существует
    if not user:                                             # если нет
        raise HTTPException(status_code=404, detail="User not found")  # 404
    db.add(Endorsement(user_id=user.id, skill_name=skill_name.strip(), from_team=(from_team or "").strip()))  # добавляем запись эндорсмента
    db.commit()                                              # фиксируем транзакцию
    return {"status": "ok"}                                  # отдаём короткий ответ

@app.post("/users/{user_id}/microstep", response_model=dict) # добавить микрошаг (для стрика)
def add_microstep(user_id: int, done_on: Optional[date] = Body(None, embed=True), db: Session = Depends(get_db)):  # дата опциональна
    user = db.get(User, user_id)                             # ищем пользователя
    if not user:                                             # если не найден
        raise HTTPException(status_code=404, detail="User not found")  # 404
    d = done_on or date.today()                              # по умолчанию — сегодняшняя дата
    try:                                                     # пробуем сохранить
        db.add(Microstep(user_id=user.id, done_on=d))        # создаём запись микрошагa
        db.commit()                                          # коммитим
    except Exception:                                        # если нарушение уникальности (дубликат дня)
        db.rollback()                                        # откатываем
        raise HTTPException(status_code=409, detail="Microstep already exists for this day")  # возвращаем 409
    return {"status": "ok"}                                  # успешный ответ

# ============================== ЛИЧНЫЙ КАБИНЕТ ================================
@app.get("/users/{user_id}/dashboard", response_model=DashboardResponse)  # собрать данные личного кабинета
def get_dashboard(user_id: int, db: Session = Depends(get_db)):           # зависимость на БД
    user = db.get(User, user_id)                             # загружаем пользователя
    if not user:                                             # если нет
        raise HTTPException(status_code=404, detail="User not found")      # 404
    total_xp = calculate_and_issue_achievements(db, user)    # пересчитываем ачивки/XP
    db.commit()                                              # фиксируем возможные новые ачивки
    progress = profile_progress_percent(user)                # считаем процент заполнения профиля
    recs = recommend_achievements(user)                      # формируем рекомендации по ачивкам
    tips = None                                              # по умолчанию советов ИИ нет
    client = scibox_client()                                 # берём клиента Scibox (если ключ задан)
    if client:                                               # если клиент доступен
        try:                                                 # пробуем получить короткие советы от LLM
            tips = client.chat.completions.create(           # вызываем чат-комплишн
                model="Qwen2.5-72B-Instruct-AWQ",            # модель Scibox
                messages=[{"role": "user", "content":
                    f"Краткие советы улучшения профиля. Роль={user.position}, отдел={user.department}, "
                    f"навыков={len(user.skills)}, проектов={len(user.projects)}. Сфокусируйся на достижениях и шагах на 2 недели."
                }],
                temperature=0.5, top_p=0.9, max_tokens=300   # параметры генерации
            ).choices[0].message.content                     # извлекаем текст ответа
        except Exception:                                    # если вызов упал
            tips = None                                      # просто скрываем советы
    ach_public = [                                           # подготавливаем список ачивок в публичном виде
        AchievementPublic(code=a.code, title=ACHIEVEMENTS_CATALOG.get(a.code, {}).get("title", a.code),
                          level=a.level, xp=a.xp, obtained_at=a.obtained_at)
        for a in user.achievements
    ]
    return DashboardResponse(                                 # собираем ответ кабинета
        user=user, progress_percent=progress, total_xp=total_xp,
        achievements=ach_public, recommended_achievements=recs, llm_tips=tips
    )

@app.get("/achievements/catalog", response_model=Dict[str, Dict[str, Any]])  # отдать каталог ачивок фронту
def get_achievements_catalog() -> Dict[str, Dict[str, Any]]:  # сигнатура с типами
    return ACHIEVEMENTS_CATALOG                               # просто возвращаем словарь

# ============================== ИИ-КОНСУЛЬТАНТ: КУРСЫ ==========================
COURSE_CATALOG = [                                           # простой внутренний каталог курсов (пример)
    {"id": "pm-101", "title": "Управление проектами: базовый", "skills": ["Управление проектами", "Коммуникации"], "provider": "PROMIS.Academy"},  # курс 1
    {"id": "py-boot", "title": "Python с нуля", "skills": ["Python", "Алгоритмы"], "provider": "PROMIS.Academy"},                                     # курс 2
    {"id": "ds-ml", "title": "ML для аналитиков", "skills": ["ML", "Python", "Данные"], "provider": "PROMIS.Academy"},                               # курс 3
    {"id": "sales-str", "title": "Стратегии продаж", "skills": ["Продажи", "Переговоры"], "provider": "PROMIS.Academy"},                             # курс 4
    {"id": "pm-adv", "title": "Project Management Advanced", "skills": ["Управление проектами", "Риски", "Стейкхолдеры"], "provider": "PROMIS.Academy"},  # курс 5
    {"id": "soft-com", "title": "Коммуникации и командная работа", "skills": ["Коммуникации", "Soft Skills"], "provider": "PROMIS.Academy"},        # курс 6
]

# ============================== LANGGRAPH: СОСТОЯНИЕ И УЗЛЫ ====================
class ChatState(TypedDict):                                   # типизированное состояние для графа
    user_id: int                                              # идентификатор пользователя
    message: str                                              # входящее сообщение от пользователя
    profile: Dict[str, Any]                                   # агрегированный профиль (роль/отдел/навыки/проекты/резюме)
    rec_courses: List[Dict[str, Any]]                         # персональные курсы (топ-3)
    llm_reply: str                                            # финальный ответ ассистента

def node_load_profile(state: ChatState, db: Session) -> Dict[str, Any]:  # узел 1: загрузка профиля
    user = db.get(User, state["user_id"])                     # читаем пользователя из БД
    if not user:                                              # если не найден
        raise HTTPException(status_code=404, detail="User not found")  # возвращаем 404
    skills = [s.name for s in user.skills]                    # собираем названия навыков
    projects = [{"title": p.title, "role": p.role, "kpi": p.result_kpi} for p in user.projects]  # собираем проекты
    profile = {                                               # агрегируем профиль для LLM
        "full_name": user.full_name,                          # ФИО
        "role": user.position,                                # должность
        "department": user.department,                        # подразделение
        "skills": skills,                                     # список навыков
        "resume": (user.resume_text or ""),                   # резюме (строка; защита от None)
        "projects": projects                                  # список проектов
    }
    return {"profile": profile}                               # возвращаем часть стейта для следующего узла

def _score_course(profile_skills: List[str], course_skills: List[str]) -> int:  # вспомогательный скоринг курса
    overlap = len(set(s.lower() for s in profile_skills) & set(cs.lower() for cs in course_skills))  # пересечение навыков
    return len(course_skills) - overlap                        # чем больше недостаёт навыков, тем выше приоритет

def node_personalize_courses(state: ChatState) -> Dict[str, Any]:  # узел 2: персонализация курсов
    prof = state["profile"]                                    # берём профиль из состояния
    skills = prof.get("skills", [])                            # получаем список навыков
    scored = []                                                # список (скор, курс)
    for c in COURSE_CATALOG:                                   # проходим весь каталог
        scored.append((_score_course(skills, c["skills"]), c)) # считаем скоринг и добавляем пару
    scored.sort(key=lambda x: x[0], reverse=True)              # сортируем по убыванию (больше gap — выше)
    rec = [x[1] for x in scored[:3]]                           # берём топ-3 курса
    return {"rec_courses": rec}                                # возвращаем персональные курсы


def node_llm_reply(state: ChatState) -> Dict[str, Any]:
    """Узел 3: генерация ответа LLM с улучшенным обработчиком ошибок"""
    client = scibox_client()  # создаём клиента Scibox (или None)
    prof = state["profile"]  # профиль
    courses = state["rec_courses"]  # подобранные курсы

    # Формируем человекочитаемый список курсов
    course_lines = "\n".join([
        f"- {c['title']} ({c['provider']}) — фокус: {', '.join(c['skills'])}" for c in courses
    ])

    # Формируем промпт для модели
    base_prompt = (
        "Ты — корпоративный ИИ-карьерный консультант. Дай персональные рекомендации, "
        "выяви пробелы компетенций и предложи 2-недельный план (микрошаги по 30–60 минут). Пиши кратко, пунктами.\n\n"
        f"Профиль: роль={prof.get('role')}, отдел={prof.get('department')}, навыки={', '.join(prof.get('skills', []))}.\n"
        f"Проекты: {', '.join(p['title'] for p in prof.get('projects', [])) or '—'}.\n"
        f"Резюме (кратко): {(prof.get('resume') or '')[:300]}.\n\n"
        f"Персональные курсы (под возможные пробелы):\n{course_lines}\n\n"
        f"Вопрос пользователя: {state['message']}\n\n"
        "Ответь: 1) Роли/возможности внутри компании; 2) Топ-курсы из списка и почему; "
        "3) Отсутствующие компетенции; 4) Пошаговый план на 2 недели с метриками прогресса."
    )

    # Если клиент не создан (нет API ключа)
    if not client:
        print("Scibox клиент не инициализирован. Проверьте API ключ и URL.")
        return {
            "llm_reply": "Курсы подобраны. Начните с №1, затем №2. Пробелы: KPI, риски, коммуникации. "
                         "План на 2 недели: выполнить вводные модули, описать 3 KPI, оформить risk-log, "
                         "подготовить апдейт стейкхолдерам, пройти практикум и внедрить метрики."
        }

    try:
        # Пробуем вызвать модель
        print(f"Отправка запроса к Scibox API с промптом длиной {len(base_prompt)} символов")

        resp = client.chat.completions.create(
            model="Qwen2.5-72B-Instruct-AWQ",
            messages=[{"role": "user", "content": base_prompt}],
            temperature=0.3,
            top_p=0.9,
            max_tokens=700
        )

        print("Успешно получен ответ от Scibox API")
        return {"llm_reply": resp.choices[0].message.content}

    except Exception as e:
        # Детальное логирование ошибки
        error_msg = f"Ошибка при обращении к Scibox API: {type(e).__name__}: {str(e)}"
        print(error_msg)

        # Логируем первые 200 символов промпта для отладки
        print(f"Промпт (первые 200 символов): {base_prompt[:200]}...")

        # Логируем информацию о клиенте (без ключа)
        print(f"Scibox base URL: {SCIBOX_BASE_URL}")
        print(f"Scibox API key: {'установлен' if SCIBOX_API_KEY else 'не установлен'}")

        return {
            "llm_reply": "Не удалось получить ответ от LLM. Используйте предложенную подборку курсов и начните с самого релевантного."
        }

def node_save_history(state: ChatState, db: Session) -> Dict[str, Any]:  # узел 4: логируем диалог
    db.add(ChatMessage(user_id=state["user_id"], role="user", content=state["message"]))      # сохраняем реплику пользователя
    db.add(ChatMessage(user_id=state["user_id"], role="assistant", content=state["llm_reply"]))  # сохраняем ответ ассистента
    db.commit()                                                # коммитим транзакцию
    return {}                                                  # узел не меняет состояние

def build_graph(db: Session) -> StateGraph:                    # сборка графа для конкретной сессии БД
    graph = StateGraph(ChatState)                              # создаём граф с типизированным состоянием
    def _load_profile(state: ChatState) -> Dict[str, Any]:     # обёртка для доступа к db внутри узла
        return node_load_profile(state, db)                    # вызываем узел загрузки профиля
    def _save(state: ChatState) -> Dict[str, Any]:             # обёртка для сохранения истории
        return node_save_history(state, db)                    # вызываем узел записи истории
    graph.add_node("load_profile", _load_profile)              # регистрируем узел загрузки профиля
    graph.add_node("personalize", node_personalize_courses)    # регистрируем узел персонализации курсов
    graph.add_node("llm", node_llm_reply)                      # регистрируем узел вызова LLM
    graph.add_node("save", _save)                              # регистрируем узел сохранения истории
    graph.set_entry_point("load_profile")                      # входная точка графа — загрузка профиля
    graph.add_edge("load_profile", "personalize")              # ребро: профиль -> персонализация
    graph.add_edge("personalize", "llm")                       # ребро: персонализация -> LLM
    graph.add_edge("llm", "save")                              # ребро: LLM -> сохранение
    graph.add_edge("save", END)                                # ребро: сохранение -> завершение
    return graph                                               # возвращаем собранный граф

# ============================== API ИИ-КОНСУЛЬТАНТА ============================
class ChatRequest(BaseModel):                                  # вход для чата
    user_id: int                                               # id пользователя
    message: str                                               # текст сообщения

class ChatResponse(BaseModel):                                 # выход для чата
    reply: str                                                 # ответ ассистента
    courses: List[Dict[str, Any]]                              # список рекомендованных курсов

@app.post("/ai/consultant/chat", response_model=ChatResponse)  # endpoint чата
def ai_consultant_chat(payload: ChatRequest, db: Session = Depends(get_db)):  # зависимость на БД
    init_state: ChatState = {"user_id": payload.user_id, "message": payload.message, "profile": {}, "rec_courses": [], "llm_reply": ""}  # стартовое состояние
    graph = build_graph(db)                                    # строим граф
    app_graph = graph.compile()                                # компилируем в исполняемую машину
    final_state: Dict[str, Any] = app_graph.invoke(init_state) # запускаем граф синхронно и получаем финальное состояние
    return ChatResponse(reply=final_state["llm_reply"], courses=final_state["rec_courses"])  # формируем ответ фронту

# ============================== ХЭЛСЧЕК ========================================
@app.get("/health", response_model=dict)                       # простой health endpoint
def health():                                                  # обработчик health
    return {"status": "ok", "time": datetime.utcnow().isoformat()}  # отдаём статус и текущий UTC
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)