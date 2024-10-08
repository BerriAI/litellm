# Base image for building
ARG LITELLM_BUILD_IMAGE=python:3.11.8-slim

# Runtime image
ARG LITELLM_RUNTIME_IMAGE=python:3.11.8-slim
# Builder stage
FROM $LITELLM_BUILD_IMAGE AS builder

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

# Build Admin UI
RUN chmod +x docker/build_admin_ui.sh && ./docker/build_admin_ui.sh

# Build the package
RUN rm -rf dist/* && python -m build

# There should be only one wheel file now, assume the build only creates one
RUN ls -1 dist/*.whl | head -1

# Install the package
RUN pip install dist/*.whl

# install dependencies as wheels
RUN pip wheel --no-cache-dir --wheel-dir=/wheels/ -r requirements.txt

# install semantic-cache [Experimental]- we need this here and not in requirements.txt because redisvl pins to pydantic 1.0 
RUN pip install redisvl==0.0.7 --no-deps

# ensure pyjwt is used, not jwt
RUN pip uninstall jwt -y
RUN pip uninstall PyJWT -y
RUN pip install PyJWT==2.9.0 --no-cache-dir

# Build Admin UI
RUN chmod +x docker/build_admin_ui.sh && ./docker/build_admin_ui.sh

# Runtime stage
FROM $LITELLM_RUNTIME_IMAGE AS runtime

# Update dependencies and clean up - handles debian security issue
RUN apt-get update && apt-get upgrade -y && rm -rf /var/lib/apt/lists/* 

WORKDIR /app
# Copy the current directory contents into the container at /app
COPY . .
RUN ls -la /app

# Copy the built wheel from the builder stage to the runtime stage; assumes only one wheel file is present
COPY --from=builder /app/dist/*.whl .
COPY --from=builder /wheels/ /wheels/

# Install the built wheel using pip; again using a wildcard if it's the only file
RUN pip install *.whl /wheels/* --no-index --find-links=/wheels/ && rm -f *.whl && rm -rf /wheels

# Generate prisma client
RUN prisma generate
RUN chmod +x docker/entrypoint.sh

EXPOSE 4000/tcp

ENTRYPOINT ["litellm"]

# Append "--detailed_debug" to the end of CMD to view detailed debug logs 
CMD ["--port", "4000"]
