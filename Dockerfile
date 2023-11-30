# Base image
FROM python:3.9-slim

# Install the project dependencies
RUN pip install -r /app/litellm/requirements.txt

EXPOSE 4000/tcp

# Start the litellm proxy, using the `litellm` cli command https://docs.litellm.ai/docs/simple_proxy
CMD litellm --config /app/proxy_server_config.yaml --port 4000
