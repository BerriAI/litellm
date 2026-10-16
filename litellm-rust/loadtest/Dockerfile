# One image with every load-test tool: mockllm, loadgen, genconfig, summarize, procmon.
#   docker build -t litellm-loadtest litellm-rust/loadtest
#   docker run litellm-loadtest mockllm -addr :9000
FROM golang:1.24.7-bookworm@sha256:b8bae5bd9ba9b1f89b635c91c24cc75cea335a16fb5076310f38566fc674b1ec AS build
WORKDIR /src
COPY go.mod ./
COPY cmd ./cmd
RUN CGO_ENABLED=0 go build -trimpath -ldflags="-s -w" -o /out/ ./cmd/...

FROM gcr.io/distroless/static-debian12:nonroot@sha256:afa5c872c891853ca7fcf1f12c3edb23f7eeef36189728842dd51042ff57f7ab
COPY --from=build /out/ /usr/local/bin/
COPY workloads /workloads
WORKDIR /work
ENTRYPOINT []
