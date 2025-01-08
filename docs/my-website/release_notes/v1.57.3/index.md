import Image from '@theme/IdealImage';

# v1.57.3 - New Base Docker Image

# 0 Critical/High Vulnerabilities

<Image img={require('../../img/release_notes/security.png')} />

## What changed?
- LiteLLMBase image now uses `cgr.dev/chainguard/python:latest-dev`

## Why the change?

To ensure there are 0 critical/high vulnerabilities on LiteLLM Docker Image

## Migration Guide

- If you use a custom dockerfile with litellm as a base image + `apt-get`

Instead of `apt-get` use `apk`, the base litellm image will no longer have `apt-get` installed.

**You are only impacted if you use `apt-get` in your Dockerfile**
```shell
# Use the provided base image
FROM ghcr.io/berriai/litellm:main-latest

# Set the working directory
WORKDIR /app

# Install dependencies - CHANGE THIS to `apk`
RUN apt-get update && apt-get install -y dumb-init 
```


Before Change
```
RUN apt-get update && apt-get install -y dumb-init
```

After Change
```
RUN apk update && apk add --no-cache dumb-init
```






