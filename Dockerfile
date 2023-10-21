FROM python:3.10

ENV LITELLM_CONFIG_PATH="/litellm.secrets.toml"
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt

WORKDIR /app/litellm-proxy 
EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "${PORT}"]