FROM python:3.13-slim

RUN pip install --no-cache-dir mcp==1.26.0

WORKDIR /app
COPY stub_server.py /app/stub_server.py

EXPOSE 8765
CMD ["python", "stub_server.py"]
