from fastapi import FastAPI, UploadFile, File
from qa.chain import get_answer_from_query

app = FastAPI()

@app.post("/ask")
async def ask_question(question: str):
    answer = get_answer_from_query(question)
    return {"answer": answer}