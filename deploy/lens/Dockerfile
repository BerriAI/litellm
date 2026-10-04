FROM python:3.12-slim
ARG LITELLM_RELEASE_TAG=""
RUN : "${LITELLM_RELEASE_TAG:?Pass --build-arg LITELLM_RELEASE_TAG matching the gateway}"
ENV LITELLM_RELEASE_TAG=${LITELLM_RELEASE_TAG}
WORKDIR /app
RUN pip install --no-cache-dir httpx==0.28.1 pydantic==2.11.7
COPY litellm/proxy/lens/__init__.py litellm/proxy/lens/models.py litellm/proxy/lens/trace_store.py litellm/proxy/lens/analysis.py litellm/proxy/lens/worker.py litellm/proxy/lens/release.py /app/lens/
COPY litellm/proxy/lens/prompts/ /app/lens/prompts/
USER 65532:65532
CMD ["python", "-m", "lens.worker"]
