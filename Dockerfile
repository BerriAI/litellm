# Base image
ARG LITELLM_BASE_IMAGE=python:3.9

# Runtime image
ARG LITELLM_RUNTIME_IMAGE=python:3.9-slim

# allow users to specify, else use python 3.9
FROM $LITELLM_BASE_IMAGE as builder

# Set the working directory to /app
WORKDIR /app

# Install build dependencies
RUN apt-get update && \
    apt-get install -y gcc python3-dev && \
    rm -rf /var/lib/apt/lists/*

# Copy the current directory contents into the container at /app
COPY requirements.txt .

# Make a virtualenv that we can copy across multistage builds
ENV VIRTUAL_ENV=/venv
RUN python -m venv $VIRTUAL_ENV
# "Activate" the virtualenv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

###############################################################################
FROM $LITELLM_RUNTIME_IMAGE as runtime

WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . .

# "Activate" the virtualenv
ENV VIRTUAL_ENV=/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

COPY --from=builder /venv /venv

EXPOSE 4000/tcp

# Start the litellm proxy, using the `litellm` cli command https://docs.litellm.ai/docs/simple_proxy
# Start the litellm proxy with default options
CMD ["--port", "4000"]

# Allow users to override the CMD when running the container, allows users to pass litellm args 
ENTRYPOINT ["litellm"]
