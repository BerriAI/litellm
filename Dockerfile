# Base image
FROM python:3.9-slim

# Copy the project files to the working directory
COPY litellm /app/litellm

# Set the working directory
WORKDIR /app/litellm

# Install the project dependencies
COPY requirements.txt /app/litellm/requirements.txt
RUN pip install -r requirements.txt

EXPOSE 4000/tcp

# Start the litellm proxy, using the `litellm` cli command https://docs.litellm.ai/docs/simple_proxy
CMD litellm --config /app/proxy_server_config.yaml --port 4000
