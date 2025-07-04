FROM python:3.14.0a3-slim

COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt

EXPOSE 3000

CMD litellm --host 0.0.0.0 --port 3000 --workers 10 --config config.yaml