# Базовый образ с Python
FROM python:3.11-slim

# Устанавливаем системные зависимости для OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Рабочая директория
WORKDIR /app

# Копируем зависимости и устанавливаем их
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем весь код проекта
COPY . .

# Создаем папки, если их нет
RUN mkdir -p templates static

# Открываем порт (Hugging Face использует 7860)
ENV PORT=7860
EXPOSE 7860

# Запуск сервера. Привязываемся к 0.0.0.0 и порту 7860
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
