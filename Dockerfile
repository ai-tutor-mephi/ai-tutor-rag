FROM python:3.11-slim

WoRKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
CMD ["python", "-m", "uvicorn", "app", "--host", "0.0.0.0", "--port", "8000"]
