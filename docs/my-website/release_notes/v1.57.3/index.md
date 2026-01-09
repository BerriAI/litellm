---
title: v1.57.3 - New Base Docker Image
slug: v1.57.3
date: 2025-01-08T10:00:00
authors:
  - name: Krrish Dholakia
    title: CEO, LiteLLM
    url: https://www.linkedin.com/in/krish-d/
    image_url: https://media.licdn.com/dms/image/v2/D4D03AQGrlsJ3aqpHmQ/profile-displayphoto-shrink_400_400/B4DZSAzgP7HYAg-/0/1737327772964?e=1749686400&v=beta&t=Hkl3U8Ps0VtvNxX0BNNq24b4dtX5wQaPFp6oiKCIHD8
  - name: Ishaan Jaffer
    title: CTO, LiteLLM
    url: https://www.linkedin.com/in/reffajnaahsi/
    image_url: https://media.licdn.com/dms/image/v2/D4D03AQGiM7ZrUwqu_Q/profile-displayphoto-shrink_800_800/profile-displayphoto-shrink_800_800/0/1675971026692?e=1741824000&v=beta&t=eQnRdXPJo4eiINWTZARoYTfqh064pgZ-E21pQTSy8jc
tags: [docker image, security, vulnerability]
hide_table_of_contents: false
---

import Image from '@theme/IdealImage';

`docker image`, `security`, `vulnerability`

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
FROM docker.litellm.ai/berriai/litellm:main-latest

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






