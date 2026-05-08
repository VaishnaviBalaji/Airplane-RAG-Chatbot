FROM python:3.11-slim

WORKDIR /app

# Install deps first — this layer is cached until requirements.txt changes
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

# Default: run the FastAPI API server
# Override with: docker run ... chainlit run chat.py --host 0.0.0.0 --port 8080
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
