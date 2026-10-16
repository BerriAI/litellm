# Rust gateway load tests

Out-of-process load testing for `litellm-gateway` (`crates/gateway`): a mock LLM, an open-loop
load generator and the gateway each run in their own container, locally or on separate cloud
machines.

| Tool | What it does |
| --- | --- |
| `mockllm` | Fake OpenAI chat completions, Anthropic messages and Bedrock invoke upstream. Streams or not, and takes its timing from a directive in the prompt (`[[mock ttft_ms=800 dur_ms=300000 out=2000]]`). Counts hits per deployment (`/d/<id>/` path prefix) at `GET /stats`; `-faults "chat-0-3=429:1.0"` makes chosen deployments fail. |
| `loadgen` | Open-loop Poisson arrivals from a workload file: weighted request classes with their own endpoint, stream share, context size (10k to 1M tokens), time to first token and total time. Writes one JSON line per request. Shard across machines with `-shard i -shards n`. |
| `genconfig` | Gateway `config.yaml` with many model groups and deployments pointing at the mocks, plus `keys.txt` and `keys.sql` (rows for `LiteLLM_VerificationToken`). `-key-seed` makes every machine generate the same keys. |
| `summarize` | Merges result files and prints proxy overhead (observed minus planned time) by stage, class and context size. |
| `procmon` | Samples the gateway process's CPU, RSS, fds and TCP connections from `/proc`. |

Workloads live in `workloads/`: `smoke`, `realistic-ramp`, `long-streams`, `large-context`,
`key-churn` and `soak`.

## Build the images

```sh
scripts/build-images.sh            # fat LTO gateway, what ships
FAST=1 scripts/build-images.sh     # no LTO, several times faster to build, for iterating
PUSH=1 REGISTRY=<registry>/<prefix> scripts/build-images.sh
```

Images are tagged `customer-proxy-<sha>` so every result maps to a commit.

## Run locally (correctness, not perf numbers)

```sh
mkdir -p out
docker run --rm -v "$PWD/out:/work/out" litellm-loadtest:<tag> genconfig -out out \
  -mocks http://mockllm:9000 -keys 100
export GATEWAY_IMAGE=litellm-rust-gateway:<tag> TOOLS_IMAGE=litellm-loadtest:<tag>
docker compose up -d gateway mockllm procmon
docker compose run --rm loadgen loadgen -workload /workloads/smoke.json \
  -target http://gateway:4000 -key sk-loadtest-master -out out/smoke.jsonl
docker compose run --rm loadgen summarize out/smoke.jsonl
docker compose logs procmon
```

Or without Docker: `go build -o bin/ ./cmd/...` and run the binaries against
`cargo run --release -p litellm-gateway` with `LITELLM_CONFIG=out/config.yaml`.

## Virtual keys

The gateway accepts the master key and the virtual keys in `LiteLLM_VerificationToken`, read from
`general_settings.database_url` or `DATABASE_URL`. Load `keys.sql` into a database the proxy's
migrations have created, start the gateway with `DATABASE_URL` set, and drop `-key` so loadgen
spreads requests over `out/keys.txt`:

```sh
psql "$DATABASE_URL" -v ON_ERROR_STOP=1 -q -f out/keys.sql
DATABASE_URL=... LITELLM_CONFIG=out/config.yaml litellm-gateway &
loadgen -workload workloads/key-churn.json -target http://127.0.0.1:4000 -messages-auth x-api-key
```

`-messages-auth x-api-key` sends the key the way the Anthropic SDK does; the gateway also reads
`Authorization: Bearer` and Azure's `api-key`. Keys are cached for 60 s (unknown keys for 5 s), so
`key-churn` exercises the miss path and other workloads mostly hit the cache. With `create_postgres`
in Terraform the stack migrates the database, loads `keys.sql` and passes `DATABASE_URL` to the
gateway; leave `KEY` unset for `cloud-run.sh`. `-key sk-loadtest-master` still works and needs no
database.

## Run in AWS

`terraform/aws` creates one VPC with a cluster placement group holding the gateway machine(s),
mock machines and load generator machines, plus ECR repositories and, optionally, an RDS Postgres
for virtual keys. With `gateway_count > 1` the generators go through an internal NLB.

```sh
cd terraform/aws
cp terraform.tfvars.example terraform.tfvars   # set ssh key, your IP, image_tag
terraform init && terraform apply
# push images to the ECR repos the stack created (machines retry pulls for an hour)
aws ecr get-login-password | docker login --username AWS --password-stdin "$(terraform output -raw ecr_registry)"
cd ../.. && PUSH=1 REGISTRY="$(terraform -chdir=terraform/aws output -raw ecr_registry)/litellm-loadtest" \
  TAG=<image_tag> scripts/build-images.sh

KEY=sk-loadtest-master scripts/cloud-run.sh smoke
KEY=sk-loadtest-master scripts/cloud-run.sh realistic-ramp
terraform -chdir=terraform/aws destroy
```

`cloud-run.sh` runs a shard on every generator, copies results to `results/<run>/`, collects
gateway samples and mock hit counts, and prints the summary.

## Reading the results

- **First-byte overhead**: time to the first SSE event (streams) or first body byte (non-streaming)
  minus the planned time. This is what the proxy adds to perceived latency.
- **Total overhead**: total time minus the planned duration.
- **Max stream gap**: the longest pause between two SSE events. Compare it with the planned chunk
  interval (duration / chunks); a gap far above it means the proxy held data back.
- **Shed** (in the loadgen log): arrivals dropped because the generator hit `max_in_flight`. Any
  shedding means the generator, not the gateway, was the limit, and the run is invalid.

## Gateway metrics and drain

- `GET /metrics` on the gateway port serves Prometheus text: request count, latency and time to
  first byte by route, model group and status, `litellm_in_flight_requests`, and
  `litellm_deployment_failure_responses_total` by deployment `api_base`. Scrape it before and
  after a run to cross-check the loadgen numbers from the gateway's side. Like the Python proxy
  it needs a key by default (`curl -H "authorization: Bearer sk-loadtest-master" .../metrics`).
  Set `litellm_settings.require_auth_for_metrics_endpoint: false` to make it public
- `GET /health/readiness` turns 503 on SIGTERM. The listener closes after
  `LITELLM_UNREADY_DELAY_SECONDS` (default 0), then open requests and streams get up to
  `LITELLM_DRAIN_SECONDS` (default 900, the NLB deregistration delay) to finish

## Bedrock deployments

`genconfig` serves half of each messages group through Bedrock invoke by default
(`-bedrock-fraction 0.5`, `bedrock_fraction` in Terraform; 0 for Anthropic only). Those deployments
carry static dummy AWS keys, so every request is SigV4-signed, and mockllm answers
`/model/<id>/invoke` and `/invoke-with-response-stream` with AWS event-stream framing that the
gateway re-encodes as Anthropic SSE. Bedrock hits show up as `bedrock-<g>-<i>` in the mock's `/stats`.

