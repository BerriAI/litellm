# Autorouter sandbox gateway

A LiteLLM deployment built from this branch that you can push to and redeploy in about a minute, for iterating on the auto router without touching the real sandbox or prod

Every model that is not defined in `config.yaml` is forwarded to the upstream gateway (`UPSTREAM_LITELLM_BASE_URL`, i.e. gateway.litellm-sandbox.ai) through `litellm_proxy/*`, so this instance inherits all upstream models and provider credentials without copying any keys. Auto routers defined here (`moe-router`) pick between those upstream models and the router code running is whatever is on this branch

## Image

`Dockerfile` overlays this branch's Python source onto the published `ghcr.io/berriai/litellm:main-latest` image instead of rebuilding deps and the UI, so a build takes seconds rather than ~15 minutes. If the branch adds a new dependency, switch `BASE_IMAGE` to an image built from the root `Dockerfile`

## Run

```bash
cp deploy/autorouter-sandbox/.env.example deploy/autorouter-sandbox/.env   # fill in
docker compose -f deploy/autorouter-sandbox/docker-compose.yml up --build -d
curl localhost:4000/v1/chat/completions -H "Authorization: Bearer $LITELLM_MASTER_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"model":"moe-router","messages":[{"role":"user","content":"hi"}]}' -i | grep x-litellm-model-name
```

Without Docker: `uv run --no-sync litellm --config deploy/autorouter-sandbox/config.yaml --port 4000`

## Deploy

`.github/workflows/deploy-autorouter-sandbox.yml` builds and pushes `ghcr.io/<owner>/litellm-autorouter-sandbox:{sha,latest}` on every push to `litellm_autorouter_sandbox_deploy`. If the repo secret `AUTOROUTER_SANDBOX_KUBECONFIG_B64` is set (base64 kubeconfig; optional repo variable `AUTOROUTER_SANDBOX_NAMESPACE`), it also applies `k8s.yaml` and rolls the deployment to the new image. Create the env secret once with the command at the top of `k8s.yaml`
