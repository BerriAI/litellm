# Base image
ARG LITELLM_BUILD_IMAGE=python:3.9

# Runtime image
ARG LITELLM_RUNTIME_IMAGE=python:3.9-slim

# allow users to specify, else use python 3.9
FROM $LITELLM_BUILD_IMAGE as builder

# Set the working directory to /app
WORKDIR /app

# Install build dependencies
RUN apt-get update && \
    apt-get install -y gcc python3-dev && \
    rm -rf /var/lib/apt/lists/*

# Copy the current directory contents into the container at /app
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install wheel && \
    pip wheel --no-cache-dir --wheel-dir=/app/wheels -r requirements.txt

###############################################################################
FROM $LITELLM_RUNTIME_IMAGE as runtime

WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . .

COPY --from=builder /app/wheels /app/wheels

RUN pip install --no-index --find-links=/app/wheels -r requirements.txt

EXPOSE 4000/tcp

# Start the litellm proxy, using the `litellm` cli command https://docs.litellm.ai/docs/simple_proxy
# Start the litellm proxy with default options
CMD ["--port", "4000"]

# Allow users to override the CMD when running the container, allows users to pass litellm args 
ENTRYPOINT ["litellm"]
