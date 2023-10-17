FROM python:3.10

COPY . /app

WORKDIR /app
RUN pip install -r requirements.txt

WORKDIR /app/litellm/proxy 
EXPOSE 8000
ENTRYPOINT [ "/bin/bash", "/app/litellm/proxy/start.sh" ]
# TODO - Set up a GitHub Action to automatically create the Docker image,
#       and then we can quickly deploy the litellm proxy in the following way
#       `docker run -p 8000:8000 ghcr.io/BerriAI/litellm:v0.8.4 -v ./secrets_template.toml:/app/secrets_template.toml`