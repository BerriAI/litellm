FROM python:3.10

RUN pip install poetry

WORKDIR /app 

COPY . .

RUN pip install -r requirements.txt

WORKDIR /app/litellm/proxy 

RUN python proxy_cli.py --config -f /app/secrets_template.toml

RUN python proxy_cli.py