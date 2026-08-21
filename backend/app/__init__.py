from pathlib import Path
import requests, fitz, chromadb
from sentence_transformers import SentenceTransformer
from .config import settings

Path(settings.upload_dir).mkdir(parents=True, exist_ok=True)
Path(settings.chroma_dir).mkdir(parents=True, exist_ok=True)
_client = chromadb.PersistentClient(path=settings.chroma_dir)
_collection = _client.get_or_create_collection("study_documents")
_embedder = SentenceTransformer(settings.embedding_model)

def extract_pages(pdf_path):
    doc = fitz.open(pdf_path)
    pages = []
    for page_no, page in enumerate(doc, 1):
        text = page.get_text("text").strip()
        if text:
            pages.append((page_no, text))
    return pages

def chunks(text, size=900, overlap=150):
    text = " ".join(text.split())
    out, start = [], 0
    while start < len(text):
        end = min(len(text), start + size)
        out.append(text[start:end])
        if end == len(text):
            break
        start = max(0, end - overlap)
    return out

def index_pdf(pdf_path, user_id, document_id, filename):
    ids, docs, metas = [], [], []
    for page_no, text in extract_pages(pdf_path):
        for i, piece in enumerate(chunks(text)):
            ids.append(f"{document_id}-{page_no}-{i}")
            docs.append(piece)
            metas.append({"user_id": str(user_id), "document_id": str(document_id),
                          "filename": filename, "page": page_no})
    if docs:
        vectors = _embedder.encode(docs, normalize_embeddings=True).tolist()
        _collection.upsert(ids=ids, documents=docs, embeddings=vectors, metadatas=metas)
    return len(docs)

def retrieve(question, user_id, k=5):
    vector = _embedder.encode([question], normalize_embeddings=True).tolist()[0]
    result = _collection.query(query_embeddings=[vector], n_results=k,
                               where={"user_id": str(user_id)})
    contexts = []
    for i, doc in enumerate(result.get("documents", [[]])[0]):
        contexts.append({"text": doc, "meta": result["metadatas"][0][i]})
    return contexts

def ask_ollama(question, contexts):
    context_text = "\n\n".join(
        f"[{c['meta']['filename']} - page {c['meta']['page']}]\n{c['text']}"
        for c in contexts
    )
    prompt = (
        "You are StudyRAG AI, a study assistant.\n"
        "Answer using the supplied document context. If the context is insufficient, "
        "say so instead of inventing facts. Keep answers student-friendly.\n\n"
        f"Question:\n{question}\n\nDocument context:\n{context_text}"
    )
    r = requests.post(f"{settings.ollama_base_url}/api/generate",
                      json={"model": settings.ollama_model,
                            "prompt": prompt, "stream": False}, timeout=180)
    r.raise_for_status()
    return r.json()["response"].strip()
