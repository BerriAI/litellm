FROM python:3.10

COPY . /app
WORKDIR /app
RUN mkdir -p /root/.config/litellm/ && cp /app/secrets_template.toml /root/.config/litellm/litellm.secrets.toml
RUN pip install -r requirements.txt

WORKDIR /app/litellm/proxy 
EXPOSE 8000
ENTRYPOINT [ "python3", "proxy_cli.py" ]
# TODO - Set up a GitHub Action to automatically create the Docker image,
#       and then we can quickly deploy the litellm proxy in the following way
#       `docker run -p 8000:8000 -v ./secrets_template.toml:/root/.config/litellm/litellm.secrets.toml ghcr.io/BerriAI/litellm:v0.8.4`