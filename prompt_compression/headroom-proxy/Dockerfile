FROM python:3.12-slim-bookworm

RUN pip install --no-cache-dir fastapi uvicorn pydantic

COPY server.py /app/server.py

WORKDIR /app

EXPOSE 8787

ENV PORT=8787

CMD ["python", "server.py"]
