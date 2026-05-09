FROM python:3.11-slim

WORKDIR /app

# Install deps first — this layer is cached until requirements.txt changes
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Cloud Run injects PORT at runtime (default 8080).
# Chainlit reads it via the --port flag in the CMD below.
ENV PORT=8080
EXPOSE 8080

# Default: Chainlit chat UI (portfolio-facing).
# For the FastAPI API locally, use docker-compose which overrides this CMD.
CMD chainlit run chat.py --host 0.0.0.0 --port $PORT
