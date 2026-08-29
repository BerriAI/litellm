# syntax=docker/dockerfile:1.7

# UI container — Next.js static export served by nginx.

ARG UI_BUILD_IMAGE=node:24.19-alpine3.24@sha256:d32cdf619f63fe0471182d08996dd516c6275bb5fd31ae06e55a570bd9e1ad43
ARG NGINX_VERSION=1.31-alpine

# ---------- builder ----------
FROM ${UI_BUILD_IMAGE} AS builder

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
FROM nginx:${NGINX_VERSION} AS runtime

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
