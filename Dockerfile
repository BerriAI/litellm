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
COPY requirements.txt .

# Build the package
RUN rm -rf dist/* && python -m build

# Install the package
RUN pip install dist/*.whl

# Install any needed packages specified in requirements.txt
RUN pip install wheel && \
    pip wheel --no-cache-dir --wheel-dir=/app/wheels -r requirements.txt

# Clear out any existing builds and build the package
RUN rm -rf dist/* && python -m build

# There should be only one wheel file now, assume the build only creates one
RUN ls -1 dist/*.whl | head -1

# Runtime stage
FROM $LITELLM_RUNTIME_IMAGE as runtime

WORKDIR /app

# Depending on wheel naming patterns, use a wildcard if multiple versions are possible
# Copy the built wheel from the builder stage to the runtime stage; assumes only one wheel file is present
COPY --from=builder /app/dist/*.whl .

# Install the built wheel using pip; again using a wildcard if it's the only file
RUN pip install *.whl && rm -f *.whl

EXPOSE 4000/tcp

# Set your entrypoint and command
ENTRYPOINT ["litellm"]
CMD ["--port", "4000"]