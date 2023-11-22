# Base image
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Copy the project files to the working directory
COPY . /app

# Install the project dependencies
RUN pip install -r requirements.txt

WORKDIR /app/litellm/proxy

# Set the command to run when the container starts
CMD python3 proxy_cli.py --config hosted_config.yaml --port 4000