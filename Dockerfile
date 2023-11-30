# Base image
FROM python:3.9-slim

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 4000/tcp

# Start the litellm proxy, using the `litellm` cli command https://docs.litellm.ai/docs/simple_proxy
CMD litellm --config /app/proxy_server_config.yaml --port 4000
