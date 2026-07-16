FROM python:3.11-slim

# Системные зависимости для OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Установка Python-зависимостей
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копирование кода
COPY . .

# Создаем папки, если их нет
RUN mkdir -p templates static

# Fly.io сам пробрасывает порт, но мы явно указываем 8000
EXPOSE 8000

# Запуск FastAPI. host 0.0.0.0 обязателен для Docker!
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
