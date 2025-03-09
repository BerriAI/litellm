FROM cgr.dev/chainguard/python:latest-dev

USER root
WORKDIR /app

ENV HOME=/home/litellm
ENV PATH="${HOME}/venv/bin:$PATH"

# Install runtime dependencies
RUN apk update && \
    apk add --no-cache gcc python3-dev openssl openssl-dev

RUN python -m venv ${HOME}/venv
RUN ${HOME}/venv/bin/pip install --no-cache-dir --upgrade pip

COPY requirements.txt .
RUN --mount=type=cache,target=${HOME}/.cache/pip \
    ${HOME}/venv/bin/pip install -r requirements.txt

EXPOSE 4000/tcp

ENTRYPOINT ["litellm"]
CMD ["--port", "4000"]