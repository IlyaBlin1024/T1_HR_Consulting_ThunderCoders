from langgraph.graph import StateGraph, START, END
from typing import TypedDict
from config import SCIBOX_API_KEY
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
import fitz  # PyMuPDF
import json
import os
import numpy as np
from typing import List, Optional, Dict, Any
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime

BASE_URL = "http://176.119.5.23:4000/v1"
client = OpenAI(api_key=SCIBOX_API_KEY, base_url=BASE_URL)

# ------------------------- Storage -------------------------
PROFILES_PATH = "profiles.json"
SAVED_PATH = "saved_lists.json"

for p in (PROFILES_PATH, SAVED_PATH):
    if not os.path.exists(p):
        with open(p, "w", encoding="utf-8") as f:
            json.dump({}, f)

app = FastAPI(title="HR Profile Service")

# ------------------------- Utilities -------------------------
def read_pdf_bytes(file_bytes: bytes) -> str:
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    return "\n".join([page.get_text() for page in doc])

def save_profiles_to_file(db: dict):
    with open(PROFILES_PATH, "w", encoding="utf-8") as f:
        json.dump(db, f, ensure_ascii=False, indent=2)

def load_profiles() -> dict:
    with open(PROFILES_PATH, "r", encoding="utf-8") as f:
        return json.load(f)
    
def save_saved_list(db: dict):
    with open(SAVED_PATH, "w", encoding="utf-8") as f:
        json.dump(db, f, ensure_ascii=False, indent=2)

def load_saved_list() -> dict:
    with open(SAVED_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def embed_texts(texts: List[str]) -> List[List[float]]:
    resp = client.embeddings.create(model="bge-m3", input=texts)
    return [item.embedding for item in resp.data]

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    a = a.reshape(1, -1)
    b = b.reshape(1, -1)
    return float(cosine_similarity(a, b)[0][0])

def compute_hr_potential(profile: dict) -> float:
    skills = profile.get("skills", [])
    if not skills:
        return 30.0  # минимальная база

    avg_level = np.mean([s.get("level", 50) for s in skills])
    skill_count = len(skills)

    readiness = profile.get("readiness_score", 50) or 50
    rotation = profile.get("open_to_rotation", "unknown")

    # Потенциал = средний уровень (вес 0.5) + разнообразие (вес 0.2) + readiness (вес 0.2) + ротация (0.1)
    potential = (
        0.5 * (avg_level) +
        0.2 * min(skill_count * 10, 100) +
        0.2 * readiness +
        0.1 * (80 if rotation == "yes" else (50 if rotation == "unknown" else 20))
    )

    return round(min(100, potential), 2)


# ------------------------- LLM Prompt -------------------------
EXTRACTION_PROMPT = (
    "You are an assistant that converts a resume text into a structured JSON profile. "
    "Return ONLY valid JSON. Fields: id, name, title, location, years_experience,"
    "skills: list of {name, level(0-100), years, category, keywords}, summary (short), stack (list), desired_salary,"
    "open_to_rotation (yes/no/unknown), readiness_score (0-100), raw_text_excerpt (short)."
)

# ------------------------- API Models -------------------------
class SearchRequest(BaseModel):
    direction: Optional[str] = None
    min_experience: Optional[int] = None
    stack: Optional[List[str]] = None
    max_salary: Optional[int] = None
    top_k: Optional[int] = 5

class MatchResult(BaseModel):
    profile_id: str
    name: Optional[str]
    title: Optional[str]
    match_score: float
    skills_overlap: List[str]
    profile_summary: Optional[str]

class EmployeeReport(BaseModel):
    profile_id: str
    name: Optional[str]
    title: Optional[str]
    summary: Optional[str]
    skills: List[Dict[str, Any]]
    

# ------------------------- LangGraph Pipeline -------------------------
class ProfileState(TypedDict):
    file_bytes: bytes
    file_text: str
    profile: dict

def node_extract_text(state: ProfileState) -> ProfileState:
    text = read_pdf_bytes(state["file_bytes"])
    state["file_text"] = text
    return state

def node_text_to_profile(state: ProfileState) -> ProfileState:
    text = state["file_text"]

    # Формируем prompt для LLM
    messages = [
        {"role": "system", "content": "You are a strict JSON extractor for resumes."},
        {"role": "user", "content": EXTRACTION_PROMPT + "\n\nResume text:\n" + text}
    ]

    # Запрос к LLM
    resp = client.chat.completions.create(
        model="Qwen2.5-72B-Instruct-AWQ",
        messages=messages,
        temperature=0.0,
        max_tokens=4000
    )
    llm_text = resp.choices[0].message.content.strip()

    try:
        start = llm_text.find("{")
        end = llm_text.rfind("}") + 1
        llm_text = llm_text[start:end]
        profile = json.loads(llm_text)
    except Exception as e:
        print("LLM JSON parse failed:", e)
        profile = {
            "name": None,
            "title": None,
            "skills": [],
            "summary": text[:200],
            "raw_text": text
        }

    # Генерируем уникальный ID всегда сами
    profile_id = f"p_{int(datetime.utcnow().timestamp()*1000)}"
    profile["id"] = profile_id

    # Создаём embedding для поиска
    embed_source = (
        " ".join([s.get('name','') for s in profile.get('skills', [])]) + "\n" + profile.get('summary','')
    )
    embedding = embed_texts([embed_source])[0]

    profile['_embedding'] = embedding
    profile['_raw_text'] = text[:1000]
    profile['_inserted_at'] = datetime.utcnow().isoformat()

    state["profile"] = profile
    return state


def node_save_profile(state: ProfileState) -> ProfileState:
    profile = state["profile"]
    db = load_profiles()
    db[profile["id"]] = profile
    save_profiles_to_file(db)
    return state

graph = StateGraph(ProfileState)
graph.add_node("extract_text", node_extract_text)
graph.add_node("convert_to_profile", node_text_to_profile)
graph.add_node("save_profile", node_save_profile)
graph.add_edge(START, "extract_text")
graph.add_edge("extract_text", "convert_to_profile")
graph.add_edge("convert_to_profile", "save_profile")
graph.add_edge("save_profile", END)
compiled_graph = graph.compile()

# ------------------------- API Endpoints -------------------------

@app.post("/ingest_pdf/")
async def ingest_pdf(file: UploadFile = File(...)):
    content = await file.read()
    initial_state = {"file_bytes": content, "file_text": "", "profile": None}
    result_state = compiled_graph.invoke(initial_state)
    profile = result_state["profile"] 
    return {"status": "ok", "profile_id": profile["id"]}



@app.post("/search_employees/", response_model=List[MatchResult])
def search_employees(req: SearchRequest):
    db = load_profiles()
    if not db:
        return []

    # Формируем embedding запроса
    qtext = " ".join(req.stack) if req.stack else (req.direction or "general")
    qemb = np.array(embed_texts([qtext])[0])

    results = []
    for pid, p in db.items():
        # Фильтры
        if req.min_experience and p.get('years_experience') is not None:
            if p['years_experience'] < req.min_experience:
                continue
        if req.max_salary and p.get('desired_salary'):
            try:
                if int(p['desired_salary']) > int(req.max_salary):
                    continue
            except Exception:
                pass
        if req.stack:
            if not any(s.lower() in ' '.join(p.get('stack', [])).lower() for s in req.stack):
                continue

        # --- Semantic similarity ---
        emb = np.array(p.get('_embedding'))
        sim = cosine_sim(qemb, emb) 

        # --- Overlap по стеку ---
        overlap = []
        if req.stack:
            pstack_lower = [s.lower() for s in p.get('stack', []) if isinstance(s, str)]
            for sk in req.stack:
                if sk.lower() in pstack_lower:
                    overlap.append(sk)
        overlap_ratio = len(overlap) / len(req.stack) if req.stack else 0  # 0..1

        # --- Взвешенная формула ---
        sim_weight = 0.1
        overlap_weight = 1.9
        score = sim_weight * sim + overlap_weight * overlap_ratio
        final_score = round(min(score / 2, 1.0) * 100, 2)

        results.append((final_score, pid, p, overlap))

    # Сортировка и top_k
    results.sort(key=lambda x: x[0], reverse=True)
    out = []
    for score, pid, p, overlap in results[:req.top_k]:
        out.append(MatchResult(
            profile_id=pid,
            name=p.get('name'),
            title=p.get('title'),
            match_score=score,
            skills_overlap=overlap,
            profile_summary=p.get('summary')
        ))

    return out

@app.post("/save_for_later/{hr_user}/{profile_id}")
def save_for_later(hr_user: str, profile_id: str):
    db = load_profiles()
    if profile_id not in db:
        raise HTTPException(status_code=404, detail="Profile not found")

    sl = load_saved_list()
    if hr_user not in sl:
        sl[hr_user] = []

    if profile_id not in sl[hr_user]:
        sl[hr_user].append(profile_id)

    save_saved_list(sl)
    return {"status": "saved", "profile_id": profile_id}


@app.get("/saved/{hr_user}")
def get_saved(hr_user: str):
    sl = load_saved_list()
    db = load_profiles()
    saved_ids = sl.get(hr_user, [])
    return [db[i] for i in saved_ids if i in db]


@app.post("/search_saved_employees/", response_model=List[MatchResult])
def search_saved_employees(hr_user: str, req: SearchRequest):
    """
    Поиск сотрудников только среди отложенных кандидатов HR-пользователя
    с фильтрами: стек, минимальный опыт, максимальная ЗП, топ K.
    """
    sl = load_saved_list()
    saved_ids = sl.get(hr_user, [])
    if not saved_ids:
        return []

    db = load_profiles()
    # Отбираем только сохранённые профили
    saved_profiles = {pid: db[pid] for pid in saved_ids if pid in db}
    if not saved_profiles:
        return []

    # Формируем embedding запроса
    qtext = " ".join(req.stack) if req.stack else (req.direction or "general")
    qemb = np.array(embed_texts([qtext])[0])

    results = []
    for pid, p in saved_profiles.items():
        # --- Фильтры ---
        if req.min_experience and p.get('years_experience') is not None:
            if p['years_experience'] < req.min_experience:
                continue
        if req.max_salary and p.get('desired_salary'):
            try:
                if int(p['desired_salary']) > int(req.max_salary):
                    continue
            except Exception:
                pass
        if req.stack:
            # проверяем пересечение стека
            if not any(s.lower() in [x.lower() for x in p.get('stack', [])] for s in req.stack):
                continue

        # --- Semantic similarity ---
        emb = np.array(p.get('_embedding'))
        sim = cosine_sim(qemb, emb)

        # --- Overlap по стеку ---
        overlap = []
        if req.stack:
            pstack_lower = [s.lower() for s in p.get('stack', []) if isinstance(s, str)]
            for sk in req.stack:
                if sk.lower() in pstack_lower:
                    overlap.append(sk)
        overlap_ratio = len(overlap) / len(req.stack) if req.stack else 0  # 0..1

        # --- Взвешенная формула ---
        sim_weight = 0.1
        overlap_weight = 1.9
        score = sim_weight * sim + overlap_weight * overlap_ratio
        final_score = round(min(score / 2, 1.0) * 100, 2)

        results.append((final_score, pid, p, overlap))

    # Сортировка и top_k
    results.sort(key=lambda x: x[0], reverse=True)
    out = []
    for score, pid, p, overlap in results[:req.top_k]:
        out.append(MatchResult(
            profile_id=pid,
            name=p.get('name'),
            title=p.get('title'),
            match_score=score,
            skills_overlap=overlap,
            profile_summary=p.get('summary')
        ))

    return out



@app.get("/employee_report/{profile_id}")
def employee_report(profile_id: str):
    db = load_profiles()
    profile = db.get(profile_id)
    if not profile:
        raise HTTPException(status_code=404, detail="Profile not found")

    return {
        "profile_id": profile_id,
        "name": profile.get("name"),
        "title": profile.get("title"),
        "summary": profile.get("summary"),
        "skills": profile.get("skills", []),
        "stack": profile.get("stack", []),
        "years_experience": profile.get("years_experience"),
        "desired_salary": profile.get("desired_salary"),
        "hr_potential": compute_hr_potential(profile)
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)