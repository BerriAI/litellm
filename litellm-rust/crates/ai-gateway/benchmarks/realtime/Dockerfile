# Multi-stage build: compile a static Go binary, ship it on a small base that
# still has a shell (busybox) so the parent service can stay alive with `sleep`
# while one-off jobs run the actual benchmark via the base64-arg wrapper.
FROM golang:1.26 AS build
WORKDIR /src
COPY go.mod go.sum ./
RUN go mod download
COPY main.go ./
# Static, stripped binary.
RUN CGO_ENABLED=0 GOOS=linux go build -ldflags="-s -w" -o /out/wsbench .

FROM busybox:1.36-glibc
# wss:// TLS verification needs a CA bundle — busybox has none, so copy the one
# from the Debian-based build stage.
COPY --from=build /etc/ssl/certs/ca-certificates.crt /etc/ssl/certs/ca-certificates.crt
COPY --from=build /out/wsbench /usr/local/bin/wsbench
COPY run.sh /usr/local/bin/run.sh
COPY serve.sh /usr/local/bin/serve.sh
RUN chmod +x /usr/local/bin/run.sh /usr/local/bin/serve.sh
# Two ways to run a leg, both taking a base64-encoded flag list (dodges Render's
# start-command tokenizer):
#   - run.sh   <b64>  → exec wsbench (stdout to the console; fine when stdout is
#                       reachable, e.g. local docker or a host whose job logs you
#                       can read).
#   - serve.sh <b64>  → run wsbench, tee output to /tmp/web/result.txt, then serve
#                       it over HTTP on $PORT. Use on a hosted runner whose
#                       one-off-job stdout is NOT surfaced (fetch /result.txt).
CMD ["sh", "-c", "echo wsbench image ready; sleep infinity"]
