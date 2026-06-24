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
RUN chmod +x /usr/local/bin/run.sh
# Parent service stays alive; real measurements run as one-off jobs whose
# start command is:  /bin/sh /usr/local/bin/run.sh <base64-of-flags>
# Keeping the arg count tiny dodges Render's start-command tokenizer quirk.
CMD ["sh", "-c", "echo wsbench image ready; sleep infinity"]
