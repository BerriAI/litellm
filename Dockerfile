FROM python:3.10

# Define a build argument for the config file path
ARG CONFIG_FILE

# Copy the custom config file (if provided) into the Docker image
COPY $CONFIG_FILE /app/config.yaml

COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt

WORKDIR /app/litellm_server
EXPOSE $PORT 

CMD exec uvicorn main:app --host 0.0.0.0 --port $PORT
