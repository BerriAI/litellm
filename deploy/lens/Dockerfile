FROM python:3.12-slim
WORKDIR /app
RUN pip install --no-cache-dir httpx==0.28.1 pydantic==2.11.7
COPY litellm/proxy/lens/__init__.py litellm/proxy/lens/models.py litellm/proxy/lens/trace_store.py litellm/proxy/lens/analysis.py litellm/proxy/lens/worker.py /app/lens/
USER 65532:65532
CMD ["python", "-m", "lens.worker"]
