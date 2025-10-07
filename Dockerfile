# Base image for building
ARG LITELLM_BUILD_IMAGE=python:3.11-slim
# Runtime image
ARG LITELLM_RUNTIME_IMAGE=python:3.11-slim

# ----------------------
# Builder stage
# ----------------------
FROM $LITELLM_BUILD_IMAGE AS builder

WORKDIR /app
USER root

# Install build dependencies + OpenSSL + tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    python3-dev \
    libssl-dev \
    openssl \
    ca-certificates \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install build
RUN pip install --upgrade "pip>=24.3.1" && pip install build

# Copy project files
COPY . .

# Build Admin UI
RUN chmod +x docker/build_admin_ui.sh && ./docker/build_admin_ui.sh

# Build Python wheel
RUN rm -rf dist/* && python -m build
RUN ls -1 dist/*.whl | head -1

# Install the built wheel
RUN pip install dist/*.whl

# Optional: install dependencies as wheels if requirements.txt exists
# Comment this out if no requirements.txt
# RUN pip wheel --no-cache-dir --wheel-dir=/wheels/ -r requirements.txt

# Ensure pyjwt is used, not jwt
RUN pip uninstall jwt -y || true
RUN pip uninstall PyJWT -y || true
RUN pip install PyJWT==2.9.0 --no-cache-dir

# ----------------------
# Runtime stage
# ----------------------
FROM $LITELLM_RUNTIME_IMAGE AS runtime

WORKDIR /app
USER root

# Install runtime dependencies + OpenSSL + tzdata + supervisor
RUN apt-get update && apt-get install -y --no-install-recommends \
    openssl \
    libssl-dev \
    ca-certificates \
    tzdata \
    supervisor \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade "pip>=24.3.1"

# Copy project files
COPY . .

# Copy wheel and install
COPY --from=builder /app/dist/*.whl .
# Copy wheels folder if you built them (optional)
# COPY --from=builder /wheels/ /wheels/
RUN pip install *.whl && rm -f *.whl
# Optional: install wheels from /wheels if you used them
# RUN pip install /wheels/* --no-index --find-links=/wheels/ && rm -rf /wheels

# Install semantic_router and aurelio-sdk using script
RUN chmod +x docker/install_auto_router.sh && ./docker/install_auto_router.sh

# Generate Prisma client
RUN prisma generate

# Entrypoints
RUN chmod +x docker/entrypoint.sh
RUN chmod +x docker/prod_entrypoint.sh

# Expose port
EXPOSE 4000/tcp

# Copy supervisor config
COPY docker/supervisord.conf /etc/supervisord.conf

ENTRYPOINT ["docker/prod_entrypoint.sh"]
CMD ["--port", "4000"]

# Verify OpenSSL
RUN openssl version -v
