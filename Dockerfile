FROM python:3.10

WORKDIR /app

# Copy everything (including scripts + data)
COPY . .

# Install dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir torch==2.2.1 --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# ✅ Build the FAISS index inside the image
RUN python scripts/build_index.py

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
