FROM python:3.13-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV PORT=8000
EXPOSE 8000
# Render (and most hosts) tell the app which port to use through $PORT.
CMD uvicorn app.api:app --host 0.0.0.0 --port ${PORT}
