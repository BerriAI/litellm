# Base image for building
ARG LITELLM_BUILD_IMAGE=python:3.9

# Runtime image
ARG LITELLM_RUNTIME_IMAGE=python:3.9-slim
# Builder stage
FROM $LITELLM_BUILD_IMAGE as builder

# Set the working directory to /app
WORKDIR /app

# Install build dependencies
RUN apt-get clean && apt-get update && \
    apt-get install -y gcc python3-dev && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip && \
    pip install build

# Copy the current directory contents into the container at /app
COPY . .

# Build the package
RUN rm -rf dist/* && python -m build

# There should be only one wheel file now, assume the build only creates one
RUN ls -1 dist/*.whl | head -1

# Install the package
RUN pip install dist/*.whl

# install dependencies as wheels
RUN pip wheel --no-cache-dir --wheel-dir=/wheels/ -r requirements.txt

# Runtime stage
FROM $LITELLM_RUNTIME_IMAGE as runtime

WORKDIR /app
# Copy the current directory contents into the container at /app
COPY . .
RUN ls -la /app

# Copy the built wheel from the builder stage to the runtime stage; assumes only one wheel file is present
COPY --from=builder /app/dist/*.whl .
COPY --from=builder /wheels/ /wheels/

# Install the built wheel using pip; again using a wildcard if it's the only file
RUN pip install *.whl /wheels/* --no-index --find-links=/wheels/ && rm -f *.whl && rm -rf /wheels

RUN chmod +x entrypoint.sh

EXPOSE 4000/tcp


# this allows accepting litellm args
ENTRYPOINT ["litellm", "--port", "4000"]