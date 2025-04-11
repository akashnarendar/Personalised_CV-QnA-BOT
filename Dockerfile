FROM python:3.10

WORKDIR /app

# Copy only what is needed first to leverage Docker caching
COPY requirements.txt ./
RUN pip install --upgrade pip && \
    pip install torch==2.2.1 --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Copy rest of the app (this should come AFTER installing requirements)
COPY . .

# Fail if CV is missing — enforces document presence
RUN test -s data/docs/Narendar_Punithan_CV_final_word.txt || (echo "❌ CV not found in data/docs/" && exit 1)

# Start the app
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
