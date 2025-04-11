from transformers import pipeline

qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")

def call_llm(question, docs):
    context = " ".join(docs)
    if len(context) < 50:
        return "No answer found – the context was too short."
    
    try:
        result = qa_pipeline(question=question, context=context)
        score = result.get("score", 0)
        if score < 0.4:
            return "🤔 Not confident there's an answer to that in the CV."
        return result.get("answer", "No answer found.")
    except Exception as e:
        return f"⚠️ Model error: {str(e)}"
