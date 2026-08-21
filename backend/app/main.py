from pathlib import Path
import sqlite3
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr
from .config import settings
from .db import init_db, connect
from .security import hash_password, verify_password, create_token, decode_token
from .rag import index_pdf, retrieve, ask_ollama

app = FastAPI(title="StudyRAG AI API")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173"],
                   allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

class RegisterIn(BaseModel):
    name: str
    email: EmailStr
    password: str

class LoginIn(BaseModel):
    email: EmailStr
    password: str

class ChatIn(BaseModel):
    question: str

def current_user(authorization: str = Header(default="")):
    if not authorization.startswith("Bearer "):
        raise HTTPException(401, "Login required")
    try:
        payload = decode_token(authorization[7:])
        con = connect()
        user = con.execute("SELECT * FROM users WHERE id=?",
                           (int(payload["sub"]),)).fetchone()
        con.close()
        if not user:
            raise HTTPException(401, "User not found")
        return dict(user)
    except Exception:
        raise HTTPException(401, "Invalid or expired token")

@app.on_event("startup")
def startup():
    init_db()
    con = connect()
    if not con.execute("SELECT id FROM users WHERE email=?",
                       (settings.admin_email,)).fetchone():
        con.execute("INSERT INTO users(name,email,password_hash,role) VALUES(?,?,?,?)",
                    ("Administrator", settings.admin_email,
                     hash_password(settings.admin_password), "admin"))
        con.commit()
    con.close()

@app.get("/api/health")
def health():
    return {"status": "ok", "app": settings.app_name}

@app.post("/api/auth/register")
def register(data: RegisterIn):
    if len(data.password) < 6:
        raise HTTPException(400, "Password must be at least 6 characters")
    con = connect()
    try:
        cur = con.execute("INSERT INTO users(name,email,password_hash) VALUES(?,?,?)",
                          (data.name.strip(), data.email.lower(), hash_password(data.password)))
        con.commit()
        user_id = cur.lastrowid
    except sqlite3.IntegrityError:
        raise HTTPException(409, "Email already registered")
    finally:
        con.close()
    return {"token": create_token(user_id, "user")}

@app.post("/api/auth/login")
def login(data: LoginIn):
    con = connect()
    user = con.execute("SELECT * FROM users WHERE email=?",
                       (data.email.lower(),)).fetchone()
    con.close()
    if not user or not verify_password(data.password, user["password_hash"]):
        raise HTTPException(401, "Invalid email or password")
    return {"token": create_token(user["id"], user["role"]),
            "user": {"id": user["id"], "name": user["name"],
                     "email": user["email"], "role": user["role"]}}

@app.get("/api/me")
def me(user=Depends(current_user)):
    return {k: user[k] for k in ("id","name","email","role")}

@app.post("/api/documents/upload")
async def upload(file: UploadFile = File(...), user=Depends(current_user)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are supported")
    safe_name = Path(file.filename).name
    user_dir = Path(settings.upload_dir) / str(user["id"])
    user_dir.mkdir(parents=True, exist_ok=True)
    con = connect()
    cur = con.execute("INSERT INTO documents(user_id,filename,path) VALUES(?,?,?)",
                      (user["id"], safe_name, ""))
    doc_id = cur.lastrowid
    path = user_dir / f"{doc_id}_{safe_name}"
    path.write_bytes(await file.read())
    con.execute("UPDATE documents SET path=? WHERE id=?", (str(path), doc_id))
    con.commit(); con.close()
    count = index_pdf(str(path), user["id"], doc_id, safe_name)
    return {"document_id": doc_id, "filename": safe_name, "chunks_indexed": count}

@app.get("/api/documents")
def documents(user=Depends(current_user)):
    con = connect()
    rows = con.execute("SELECT id,filename,created_at FROM documents WHERE user_id=? ORDER BY id DESC",
                       (user["id"],)).fetchall()
    con.close()
    return [dict(r) for r in rows]

@app.post("/api/chat")
def chat(data: ChatIn, user=Depends(current_user)):
    if not data.question.strip():
        raise HTTPException(400, "Question is required")
    contexts = retrieve(data.question, user["id"])
    answer = ask_ollama(data.question, contexts) if contexts else (
        "I couldn't find relevant content in your uploaded documents."
    )
    sources = [{"filename": c["meta"]["filename"], "page": c["meta"]["page"]}
               for c in contexts]
    con = connect()
    con.execute("INSERT INTO chats(user_id,question,answer) VALUES(?,?,?)",
                (user["id"], data.question, answer))
    con.commit(); con.close()
    return {"answer": answer, "sources": sources}

@app.get("/api/chats")
def chats(user=Depends(current_user)):
    con = connect()
    rows = con.execute("SELECT id,question,answer,created_at FROM chats "
                       "WHERE user_id=? ORDER BY id DESC LIMIT 50",
                       (user["id"],)).fetchall()
    con.close()
    return [dict(r) for r in rows]

@app.get("/api/admin/stats")
def admin_stats(user=Depends(current_user)):
    if user["role"] != "admin":
        raise HTTPException(403, "Admin access required")
    con = connect()
    data = {
        "users": con.execute("SELECT COUNT(*) c FROM users").fetchone()["c"],
        "documents": con.execute("SELECT COUNT(*) c FROM documents").fetchone()["c"],
        "chats": con.execute("SELECT COUNT(*) c FROM chats").fetchone()["c"],
    }
    con.close()
    return data
