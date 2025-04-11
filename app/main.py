from fastapi import FastAPI, UploadFile, File
from qa.chain import get_answer_from_query
from qa.retriever import load_index

app = FastAPI()

# ✅ Load index on app startup
@app.on_event("startup")
def startup_event():
    try:
        load_index()
        print("✅ Index loaded on startup.")
    except Exception as e:
        print(f"⚠️ Index failed to load on startup: {e}")

# ✅ Endpoint to ask questions
@app.post("/ask")
async def ask_question(question: str):
    answer = get_answer_from_query(question)
    return {"answer": answer}

# ✅ Endpoint to reload FAISS index manually (e.g. from Airflow)
@app.post("/reload-index")
async def reload_index():
    try:
        load_index(force_reload=True)
        return {"status": "success", "message": "Index reloaded"}
    except Exception as e:
        return {"status": "error", "message": str(e)}
