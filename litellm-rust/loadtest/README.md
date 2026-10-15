# Rust gateway load tests

Out-of-process load testing for `litellm-gateway` (`crates/gateway`): a mock LLM, an open-loop
load generator and the gateway each run in their own container, locally or on separate cloud
machines.

| Tool | What it does |
| --- | --- |
| `mockllm` | Fake OpenAI chat completions and Anthropic messages upstream. Streams or not, and takes its timing from a directive in the prompt (`[[mock ttft_ms=800 dur_ms=300000 out=2000]]`). Counts hits per deployment (`/d/<id>/` path prefix) at `GET /stats`; `-faults "chat-0-3=429:1.0"` makes chosen deployments fail. |
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
  -mocks http://mockllm:9000 -chat-provider openai_like -keys 100
export GATEWAY_IMAGE=litellm-rust-gateway:<tag> TOOLS_IMAGE=litellm-loadtest:<tag>
docker compose up -d gateway mockllm procmon
docker compose run --rm loadgen loadgen -workload /workloads/smoke.json \
  -target http://gateway:4000 -key sk-loadtest-master -out out/smoke.jsonl
docker compose run --rm loadgen summarize out/smoke.jsonl
docker compose logs procmon
```

Or without Docker: `go build -o bin/ ./cmd/...` and run the binaries against
`cargo run --release -p litellm-gateway` with `LITELLM_CONFIG=out/config.yaml`.

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

## Until the gateway gains these features

- Virtual keys: the Rust gateway only accepts the master key today, so pass `-key sk-loadtest-master`
  (`KEY=` for `cloud-run.sh`). Once keys are read from Postgres, drop it and set `create_postgres`.
- `/v1/messages` auth: `-messages-auth x-api-key` sends the key the way the Anthropic SDK does.
  The gateway reads only `Authorization: Bearer` today.
- Chat: deployments use `openai_like/` because the gateway rejects `openai/`, and chat streaming
  returns 501.
- Routing: deployments sharing a model name overwrite each other, so all of a group's traffic goes
  to its last deployment (visible in the mock's `/stats`).
