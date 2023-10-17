FROM python:3.10

COPY . /app

RUN pip install -r requirements.txt

WORKDIR /app/litellm/proxy 

ENTRYPOINT [ "/bin/bash", "/app/litellm/proxy/start.sh" ]