# Base image
FROM python:3.9-slim

# Copy the project files to the working directory
COPY litellm /app/litellm

# Set the working directory
WORKDIR /app/litellm

# Install the project dependencies
COPY requirements.txt /app/litellm/requirements.txt
RUN pip install -r requirements.txt

WORKDIR /app/litellm/proxy

COPY hosted_config.yaml /app/hosted_config.yaml

EXPOSE 4000/tcp

# Set the command to run when the container starts
CMD python3 proxy_cli.py --config /app/hosted_config.yaml --port 4000