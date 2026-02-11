# Base image for building
ARG LITELLM_BUILD_IMAGE=cgr.dev/chainguard/python:latest-dev

# Runtime image
ARG LITELLM_RUNTIME_IMAGE=cgr.dev/chainguard/python:latest-dev

# Builder stage
FROM $LITELLM_BUILD_IMAGE AS builder

# Set the working directory to /app
WORKDIR /app

USER root

# Install build dependencies
RUN apk add --no-cache bash gcc git openssl openssl-dev

RUN python -m pip install --upgrade pip setuptools && \
    python -m pip install build wheel cmake setuptools_rust maturin

# Configure Rust environment for building Rust-backed Python wheels
ENV CARGO_HOME=/root/.cargo
ENV PATH=/root/.cargo/bin:$PATH
# Allow forward-compatible ABI for PyO3-based packages on Python 3.14
ENV PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1

RUN apk add --no-cache curl build-base openssl-dev && \
    curl https://sh.rustup.rs -sSf | sh -s -- -y && \
    rustup default stable && \
    python -m pip install maturin

# Copy the current directory contents into the container at /app
COPY . .

# Build Admin UI
# Convert Windows line endings to Unix and make executable
RUN sed -i 's/\r$//' docker/build_admin_ui.sh && chmod +x docker/build_admin_ui.sh && ./docker/build_admin_ui.sh

# Build the package
RUN rm -rf dist/* && python -m build

# There should be only one wheel file now, assume the build only creates one
RUN ls -1 dist/*.whl | head -1

# Install the package
RUN pip install dist/*.whl

# Install build dependencies needed for git-based packages
RUN pip install --no-cache-dir hatchling hatch-vcs

# Install dependencies as wheels, force binary for Pillow/tokenizers to avoid build issues
RUN pip wheel --no-cache-dir --wheel-dir=/wheels/ --only-binary=Pillow,tokenizers -r requirements.txt
# ensure pyjwt is used, not jwt
RUN pip uninstall jwt -y
RUN pip uninstall PyJWT -y
RUN pip install PyJWT==2.9.0 --no-cache-dir

# Runtime stage
FROM $LITELLM_RUNTIME_IMAGE AS runtime

# Ensure runtime stage runs as root
USER root

# Install runtime dependencies (libsndfile needed for audio processing on ARM64)
RUN apk add --no-cache bash openssl tzdata nodejs npm libsndfile && \
    npm install -g npm@latest tar@7.5.7 glob@11.1.0 @isaacs/brace-expansion@5.0.1 && \
    # SECURITY FIX: npm bundles tar, glob, and brace-expansion at multiple nested
    # levels inside its dependency tree. `npm install -g <pkg>` only creates a
    # SEPARATE global package, it does NOT replace npm's internal copies.
    # We must find and replace EVERY copy inside npm's directory.
    GLOBAL="$(npm root -g)" && \
    find "$GLOBAL/npm" -type d -name "tar" -path "*/node_modules/tar" | while read d; do \
        rm -rf "$d" && cp -rL "$GLOBAL/tar" "$d"; \
    done && \
    find "$GLOBAL/npm" -type d -name "glob" -path "*/node_modules/glob" | while read d; do \
        rm -rf "$d" && cp -rL "$GLOBAL/glob" "$d"; \
    done && \
    find "$GLOBAL/npm" -type d -name "brace-expansion" -path "*/node_modules/@isaacs/brace-expansion" | while read d; do \
        rm -rf "$d" && cp -rL "$GLOBAL/@isaacs/brace-expansion" "$d"; \
    done && \
    npm cache clean --force

WORKDIR /app
# Copy the current directory contents into the container at /app
COPY . .
RUN ls -la /app

# Copy the built wheel from the builder stage to the runtime stage; assumes only one wheel file is present
COPY --from=builder /app/dist/*.whl .
COPY --from=builder /wheels/ /wheels/

# Install the built wheel using pip; again using a wildcard if it's the only file
RUN pip install *.whl /wheels/* --no-index --find-links=/wheels/ && rm -f *.whl && rm -rf /wheels

# Replace the nodejs-wheel-binaries bundled node with the system node (fixes CVE-2025-55130)
RUN NODEJS_WHEEL_NODE=$(find /usr/lib -path "*/nodejs_wheel/bin/node" 2>/dev/null) && \
    if [ -n "$NODEJS_WHEEL_NODE" ]; then cp /usr/bin/node "$NODEJS_WHEEL_NODE"; fi

# Remove test files and keys from dependencies
RUN find /usr/lib -type f -path "*/tornado/test/*" -delete && \
    find /usr/lib -type d -path "*/tornado/test" -delete

# SECURITY FIX: nodejs-wheel-binaries (pip package used by Prisma) bundles a complete
# npm with old vulnerable deps at /usr/lib/python3.*/site-packages/nodejs_wheel/.
# Patch every copy of tar, glob, and brace-expansion inside that tree.
RUN GLOBAL="$(npm root -g)" && \
    find /usr/lib -path "*/nodejs_wheel/*/node_modules/tar" -type d | while read d; do \
        rm -rf "$d" && cp -rL "$GLOBAL/tar" "$d"; \
    done && \
    find /usr/lib -path "*/nodejs_wheel/*/node_modules/glob" -type d | while read d; do \
        rm -rf "$d" && cp -rL "$GLOBAL/glob" "$d"; \
    done && \
    find /usr/lib -path "*/nodejs_wheel/*/node_modules/@isaacs/brace-expansion" -type d | while read d; do \
        rm -rf "$d" && cp -rL "$GLOBAL/@isaacs/brace-expansion" "$d"; \
    done

# Install semantic_router and aurelio-sdk using script
# Convert Windows line endings to Unix and make executable
RUN sed -i 's/\r$//' docker/install_auto_router.sh && chmod +x docker/install_auto_router.sh && ./docker/install_auto_router.sh

# Generate prisma client using the correct schema
RUN prisma generate --schema=./litellm/proxy/schema.prisma
# Convert Windows line endings to Unix for entrypoint scripts
RUN sed -i 's/\r$//' docker/entrypoint.sh && chmod +x docker/entrypoint.sh
RUN sed -i 's/\r$//' docker/prod_entrypoint.sh && chmod +x docker/prod_entrypoint.sh

EXPOSE 4000/tcp

RUN apk add --no-cache supervisor
COPY docker/supervisord.conf /etc/supervisord.conf

ENTRYPOINT ["docker/prod_entrypoint.sh"]

# Append "--detailed_debug" to the end of CMD to view detailed debug logs
CMD ["--port", "4000"]
