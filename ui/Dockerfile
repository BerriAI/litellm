# syntax=docker/dockerfile:1.7

# UI container — Next.js static export served by nginx.

ARG NODE_VERSION=20.18-alpine3.20
ARG NGINX_VERSION=1.27-alpine

# ---------- builder ----------
FROM node:${NODE_VERSION}@sha256:3488b10bf958af7125a176419d2d8a9937d895bf124012aae811651988d2ffe6 AS builder

ENV NEXT_TELEMETRY_DISABLED=1 \
    npm_config_fund=false \
    npm_config_audit=false

WORKDIR /app

# Layer the lockfile-only install above the source copy so source-only
# edits don't bust the install cache.
COPY ui/litellm-dashboard/package.json ui/litellm-dashboard/package-lock.json ./
RUN --mount=type=cache,target=/root/.npm \
    npm ci --prefer-offline

COPY ui/litellm-dashboard/ ./
RUN npm run build

# ---------- runtime ----------
FROM nginx:${NGINX_VERSION}@sha256:65645c7bb6a0661892a8b03b89d0743208a18dd2f3f17a54ef4b76fb8e2f2a10 AS runtime

# Drop the upstream default :80 server; we own the config.
RUN rm -f /etc/nginx/conf.d/default.conf

# Static export → web root.
COPY --from=builder /app/out /usr/share/nginx/html

# Routing rules — see ui/nginx.conf for the full description.
COPY ui/nginx.conf /etc/nginx/nginx.conf

EXPOSE 3000/tcp

# nginx as PID 1 in foreground; respects SIGTERM out of the box, so
# no tini/dumb-init wrapper needed.
CMD ["nginx", "-g", "daemon off;"]
