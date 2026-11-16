#!/bin/bash
# Creates the one task container: glibc 2.39 runtime for the CI-built native module. Host network because shared Postgres/Redis listen on 127.0.0.1.
H=/home/ubuntu; C=$H/cachezero; SO=$C/scratch/native/x/litellm/rust_bridge/_native.abi3.so; N=$H/.nvm/versions/node/v24.22.3
docker run -d --name cachezero-pr2-glibc239 --network host --user 1000:1000 --log-driver json-file --log-opt max-size=10m --log-opt max-file=3 \
 -e HOME=$H -e PYTHONDONTWRITEBYTECODE=1 -e npm_config_cache=$C/scratch/npm \
 -v $H/.pyenv/versions/3.12.16:$H/.pyenv/versions/3.12.16:ro -v $H/repos/litellm/.venv:$H/repos/litellm/.venv:ro -v $N:$N:ro \
 -v $C:$C -v $C/wt:$C/wt:ro -v $C/wt/pr2/litellm/cli:$H/repos/litellm/litellm/cli:ro \
 -v $SO:$C/wt/pr2/litellm/rust_bridge/_native.abi3.so:ro -v $SO:$C/wt/head/litellm/rust_bridge/_native.abi3.so:ro \
 -v $H/.cache/prisma-python:$H/.cache/prisma-python:ro -v $H/.cache/prisma:$H/.cache/prisma:ro \
 ubuntu@sha256:73b999393d07f2bf65a40373778a44dc9a6d54ad7718d29247d8ff9413c4d6a8 sleep infinity
docker exec -u 0 cachezero-pr2-glibc239 bash -c 'apt-get update -qq >/dev/null 2>&1; DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends libsqlite3-0 libreadline8t64 ca-certificates >/dev/null 2>&1; echo apt rc=$?'
