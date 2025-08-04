FROM python:3.10-slim

WORKDIR /app

# Copy requirements first for caching
COPY requirements.txt .

RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Copy all project files including the pickle file
COPY . .

EXPOSE 5000

CMD ["python", "app.py"]
