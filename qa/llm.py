from transformers import pipeline

qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")

def call_llm(question, docs):
    context = " ".join(docs)
    if len(context) < 50:
        return "No answer found – the context was too short.", 0.0

    try:
        result = qa_pipeline(question=question, context=context)
        return result.get("answer", "No answer found."), result.get("score", 0.0)
    except Exception as e:
        return f"⚠️ Model error: {str(e)}", 0.0
