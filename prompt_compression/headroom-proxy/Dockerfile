FROM python:3.12-slim-bookworm

RUN pip install --no-cache-dir fastapi uvicorn pydantic

COPY server.py /app/server.py

WORKDIR /app

EXPOSE 10000

ENV PORT=10000

CMD ["python", "server.py"]
