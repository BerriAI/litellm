FROM debian:bookworm-slim@sha256:3783cc01769c7b2b1b83a5c5ad96c815348e28ed7da68e2e3687004faa906251

ARG GH_VERSION=2.101.0
ARG GH_SHA256=9bca2d1c16825f109907a23307628a2f0698fbf99662b73a5cf0b020293072b8
ARG UV_VERSION=0.10.9
ARG UV_SHA256=20d79708222611fa540b5c9ed84f352bcd3937740e51aacc0f8b15b271c57594
ARG CLAUDE_CODE_VERSION=2.1.228
ARG CLAUDE_CODE_SHA256=d535985e6941a3eb00179ccd7f52ceb0c6623a0305a518ebc4e6514f84a94c99

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates curl git jq procps iproute2 \
 && rm -rf /var/lib/apt/lists/*

RUN curl -fsSLo /tmp/gh.tar.gz "https://github.com/cli/cli/releases/download/v${GH_VERSION}/gh_${GH_VERSION}_linux_amd64.tar.gz" \
 && echo "${GH_SHA256}  /tmp/gh.tar.gz" | sha256sum -c - \
 && tar -xzf /tmp/gh.tar.gz -C /usr/local/bin --strip-components=2 "gh_${GH_VERSION}_linux_amd64/bin/gh" \
 && rm /tmp/gh.tar.gz

RUN curl -fsSLo /tmp/uv.tar.gz "https://github.com/astral-sh/uv/releases/download/${UV_VERSION}/uv-x86_64-unknown-linux-gnu.tar.gz" \
 && echo "${UV_SHA256}  /tmp/uv.tar.gz" | sha256sum -c - \
 && tar -xzf /tmp/uv.tar.gz -C /usr/local/bin --strip-components=1 uv-x86_64-unknown-linux-gnu/uv \
 && rm /tmp/uv.tar.gz

RUN curl -fsSLo /tmp/claude "https://downloads.claude.ai/claude-code-releases/${CLAUDE_CODE_VERSION}/linux-x64/claude" \
 && echo "${CLAUDE_CODE_SHA256}  /tmp/claude" | sha256sum -c - \
 && install -m 0755 /tmp/claude /usr/local/bin/claude \
 && rm /tmp/claude

RUN groupadd --gid 1000 populator && useradd --uid 1000 --gid 1000 --create-home populator

ENV HOME=/home/populator \
    LITELLM_REPO=/opt/litellm \
    DISABLE_AUTOUPDATER=1

COPY --chown=populator:populator . /opt/litellm/tests/e2e/

USER populator
WORKDIR /home/populator
CMD ["/opt/litellm/tests/e2e/claude_code/cron_vm/run_daily.sh"]
