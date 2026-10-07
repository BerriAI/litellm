ARG LITELLM_BUILD_IMAGE=cgr.dev/chainguard/wolfi-base@sha256:1d95114038f76513a9ace6fca107d5582b08c65981f81f61cb56bf7fd2ef216d
ARG LITELLM_RUNTIME_IMAGE=cgr.dev/chainguard/wolfi-base@sha256:1d95114038f76513a9ace6fca107d5582b08c65981f81f61cb56bf7fd2ef216d

FROM $LITELLM_BUILD_IMAGE AS builder
RUN apk add --no-cache rust build-base cmake perl pkgconf openssl-dev libseccomp-dev python-3.13
WORKDIR /src
COPY .cargo/ .cargo/
COPY litellm-rust/ litellm-rust/
COPY model_prices_and_context_window.json model_prices_and_context_window.json
COPY litellm/integrations/otel/model/providers.json litellm/integrations/otel/model/providers.json
COPY litellm/proxy/lens/prompts/ litellm/proxy/lens/prompts/
WORKDIR /src/litellm-rust
ENV CARGO_PROFILE_RELEASE_DEBUG=0 CARGO_PROFILE_RELEASE_STRIP=symbols
RUN cargo build --locked --release -p litellm-lens
COPY deploy/lens/python_policy.c /tmp/python_policy.c
RUN cc -std=c11 -D_GNU_SOURCE -O2 -Wall -Wextra -Werror /tmp/python_policy.c -lseccomp -o /tmp/python-policy && \
    /tmp/python-policy /tmp/python.seccomp

FROM builder AS test-builder
RUN cargo test --locked --release -p litellm-lens --test sandbox --no-run --message-format=json > /tmp/test-artifacts.json && \
    python3.13 -c 'import json, pathlib, shutil; rows = [json.loads(line) for line in pathlib.Path("/tmp/test-artifacts.json").read_text().splitlines()]; artifact, = [r["executable"] for r in rows if r.get("executable") and r["target"]["name"] == "sandbox"]; shutil.copyfile(artifact, "/tmp/lens-sandbox-tests")' && \
    chmod 755 /tmp/lens-sandbox-tests

FROM $LITELLM_RUNTIME_IMAGE AS service
ARG LITELLM_RELEASE_TAG=""
RUN : "${LITELLM_RELEASE_TAG:?Pass --build-arg LITELLM_RELEASE_TAG matching the gateway}"
RUN apk add --no-cache python-3.13 setpriv libgcc libstdc++ openssl ca-certificates
ENV LITELLM_RELEASE_TAG=${LITELLM_RELEASE_TAG} PYTHONDONTWRITEBYTECODE=1
WORKDIR /app
COPY --from=builder /src/litellm-rust/target/release/litellm-lens /usr/local/bin/litellm-lens
COPY --from=builder /tmp/python.seccomp /app/lens/python.seccomp
COPY deploy/lens/python_runtime.py /tmp/python_runtime.py
RUN python3.13 -S /tmp/python_runtime.py /app/lens/python-runtime.json && rm /tmp/python_runtime.py
USER 65532:65532
EXPOSE 4318
ENTRYPOINT ["/usr/local/bin/litellm-lens"]

FROM service AS smoke
COPY --from=test-builder /tmp/lens-sandbox-tests /usr/local/bin/lens-sandbox-tests
ENTRYPOINT ["/usr/local/bin/lens-sandbox-tests"]
CMD ["--ignored", "--nocapture", "--test-threads=1"]

FROM service AS runtime
