# Base image
ARG LITELLM_BASE_IMAGE=python:3.9-slim

# allow users to specify, else use python 3.9-slim
FROM $LITELLM_BASE_IMAGE

# Set the working directory to /app
WORKDIR /app

# Install build dependencies
RUN apt-get update && \
    apt-get install -y gcc python3-dev && \
    rm -rf /var/lib/apt/lists/*

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip wheel --no-cache-dir --wheel-dir=wheels -r requirements.txt
RUN pip install --no-cache-dir --find-links=wheels -r requirements.txt

EXPOSE 4000/tcp

# Start the litellm proxy, using the `litellm` cli command https://docs.litellm.ai/docs/simple_proxy

# Start the litellm proxy with default options
CMD ["--port", "4000"]

# Allow users to override the CMD when running the container, allows users to pass litellm args 
ENTRYPOINT ["litellm"]
