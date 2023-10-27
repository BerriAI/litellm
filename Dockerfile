FROM python:3.10

COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt

WORKDIR /app/litellm_server
EXPOSE $PORT 

CMD exec uvicorn main:app --host 0.0.0.0 --port $PORT
