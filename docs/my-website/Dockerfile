FROM python:3.14.0a3-slim

COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt

EXPOSE $PORT 

CMD litellm --host 0.0.0.0 --port $PORT --workers 10 --config config.yaml