# Managed Agents

Spin up sandboxed coding agents on AWS Fargate. Each agent is a containerized harness (e.g. `opencode`) that clones a git repo, talks to LiteLLM as its model provider, and exposes an HTTP API that the proxy proxies through.

## Architecture

```
client ──► litellm proxy ──► /v1/managed_agents/* endpoints
                                │
                                ├── Postgres (templates, agents, sessions)
                                └── AWS
                                     ├── ECR  (one repo per dockerfile_id)
                                     ├── ECS  (cluster: litellm-agents)
                                     └── Fargate task per session
                                          │
                                          └─► harness (e.g. opencode) on container_port
                                                │
                                                └─► LiteLLM API (model calls)
```

Lifecycle: admin registers a **sandbox template** (dockerfile + repo). User creates an **agent** (template + model + prompt). User opens a **session** — proxy launches a Fargate task, waits for the harness HTTP server, then forwards messages to it.

## Prerequisites

- **Postgres** with the LiteLLM Prisma schema applied (the `LiteLLM_ManagedAgent*` tables).
- **AWS account** with permission to create ECR repos, ECS clusters, IAM roles, security groups, and run Fargate tasks. The proxy auto-bootstraps shared infra on first template build.
- **Default VPC** in your target region with at least one public subnet (`map-public-ip-on-launch=true`). Or pass overrides (see `aws.subnets` / `aws.security_group` below).
- **Docker** on the proxy host. The proxy shells out to `docker build` / `docker push` to publish harness images to ECR.
- **AWS CLI credentials** in the proxy's environment (env vars or `~/.aws/credentials`). The default boto3 chain is used.

## Configuration

Add a `managed_agents` block under `general_settings` in your proxy YAML:

```yaml
general_settings:
  master_key: sk-1234
  managed_agents:
    enabled: true
    aws_region: us-west-2
    dockerfiles:
      # Bundled sample harness — `path` is omitted, resolved to
      # litellm/proxy/managed_agents_endpoints/harnesses/opencode/Dockerfile
      opencode:
        container_port: 4096
      # Custom harness — `path` is required and resolved relative to the
      # proxy's working directory.
      # my-harness:
      #   path: ./managed_agents_endpoints/harnesses/my-harness/Dockerfile
      #   container_port: 4096
    aws: {}                       # leave empty to auto-discover default VPC + create resources
    reconcile_interval_seconds: 60
```

### `dockerfiles`

Each entry registers a harness image the proxy can build and run.

| Field | Required | Description |
|---|---|---|
| `path` | no | Path to the Dockerfile. Resolved relative to the proxy's working directory. The directory containing the Dockerfile is the build context. **Omit `path` to use a bundled harness** — the proxy will look for `harnesses/<dockerfile_id>/Dockerfile` next to its source files. |
| `container_port` | yes | Port the harness listens on inside the container. SG ingress is opened to `0.0.0.0/0:container_port`. |
| `build_platform` | no | Docker target platform. Default `linux/amd64`. Set `linux/arm64` to build + run on Graviton. The value is also mapped onto the ECS task def's `runtimePlatform.cpuArchitecture`, so image and Fargate task always agree on architecture. |

### `aws` overrides

Leave `aws: {}` to auto-discover and create everything. Override individual fields when you need to pin to existing infra:

| Field | Description |
|---|---|
| `cluster` | ECS cluster name (default: `litellm-agents`). Created if missing. |
| `subnets` | List of subnet IDs. Must be in the same VPC and have public IPs. If unset, the proxy picks one public subnet from the default VPC. |
| `security_group` | SG ID with inbound `container_port` and outbound 443 + DNS. If unset, the proxy creates one. |
| `task_execution_role_arn` | IAM role for ECS to pull from ECR and write CloudWatch logs. If unset, the proxy creates `litellm-agents-task-exec`. |

## Adding a harness

A harness is any container that exposes the OpenCode-compatible HTTP API on `container_port`. Drop a folder under `harnesses/<harness_id>/` containing a `Dockerfile` and any helper files (e.g. `entrypoint.sh`), then register it under `dockerfiles` in the YAML.

When the proxy launches a Fargate task for a session, it injects the following env vars into the container as `containerOverrides.environment` on the ECS `RunTask` call. The values come from the per-session agent + template rows in the DB — you do NOT set them in the Dockerfile or yaml. **The harness only needs to read them at startup** (e.g. in `entrypoint.sh`).

| Env | Set by proxy from |
|---|---|
| `REPO_URL` | Template `repo_url` |
| `BRANCH` | Agent `branch` (or template `default_branch`) |
| `LITELLM_API_KEY` | Agent `litellm_api_key` |
| `LITELLM_API_BASE` | Agent `litellm_api_base` |
| `LITELLM_DEFAULT_MODEL` | Agent `model` |
| `AGENT_PROMPT` | Agent `prompt` (optional) |
| `GIT_TOKEN` | Decrypted from template's `git_credential_id` (optional, only for `visibility=private`) |
| `PORT` | `container_port` |

Concretely: a client calls `POST /v1/managed_agents/agents` with `litellm_api_key` + `litellm_api_base` + `model` in the body — those are stored on the agent row. Later, when that agent's session is opened, the proxy reads the row and writes the values into the container's environment block before `RunTask`. Each session gets its own, isolated set of env vars.

The shipped `harnesses/opencode/` is the reference implementation.

### Build platform

The proxy passes `--platform <build_platform>` to `docker build` and matches it to the ECS task def's `runtimePlatform.cpuArchitecture`. Default is `linux/amd64`. Set `build_platform: linux/arm64` per dockerfile entry to target Graviton. Build host architecture doesn't matter — Apple Silicon hosts can build amd64 images thanks to QEMU emulation.

## End-to-end flow

```
# 1. List configured dockerfiles
curl -H "Authorization: Bearer $KEY" $PROXY/v1/managed_agents/dockerfiles

# 2. Create a template (admin only — slow first time: docker build + ECR push + register task def)
curl -H "Authorization: Bearer $KEY" -H "content-type: application/json" \
  -X POST $PROXY/v1/managed_agents/sandbox-templates \
  -d '{"name":"my-template","dockerfile_id":"opencode","repo_url":"https://github.com/owner/repo","default_branch":"main","visibility":"public"}'

# 3. Create an agent (any user)
curl -H "Authorization: Bearer $KEY" -H "content-type: application/json" \
  -X POST $PROXY/v1/managed_agents/agents \
  -d '{"name":"my-agent","model":"anthropic/claude-sonnet-4-6","prompt":"Concise coding agent.","tools":[],"litellm_api_key":"sk-...","litellm_api_base":"https://...","template_id":"<template_id>"}'

# 4. Open a session (~50–120s cold boot)
curl -H "Authorization: Bearer $KEY" -H "content-type: application/json" \
  -X POST $PROXY/v1/managed_agents/agents/$AGENT_ID/session \
  -d '{"title":"smoke","initial_prompt":"What does this repo do?"}'

# 5. Send a follow-up message
curl -H "Authorization: Bearer $KEY" -H "content-type: application/json" \
  -X POST $PROXY/v1/managed_agents/sessions/$SESSION_ID/message \
  -d '{"text":"List the top-level directories."}'

# 6. Tear down
curl -H "Authorization: Bearer $KEY" -X DELETE $PROXY/v1/managed_agents/sessions/$SESSION_ID
```

## Endpoints

| Method | Path | Auth | Description |
|---|---|---|---|
| GET | `/v1/managed_agents/dockerfiles` | any | List configured dockerfile entries. |
| POST | `/v1/managed_agents/sandbox-templates` | admin | Build + push image, register task def, persist template row. |
| GET | `/v1/managed_agents/sandbox-templates` | any | List templates. |
| GET | `/v1/managed_agents/sandbox-templates/{template_id}` | any | Fetch one template. |
| DELETE | `/v1/managed_agents/sandbox-templates/{template_id}` | admin | Delete template. Refused if any agents reference it. |
| POST | `/v1/managed_agents/agents` | any | Create an agent bound to a template. |
| GET | `/v1/managed_agents/agents/{agent_id}` | any | Fetch one agent. |
| POST | `/v1/managed_agents/agents/{agent_id}/session` | any | Launch a Fargate task, wait for ready, optionally send `initial_prompt`. |
| GET | `/v1/managed_agents/sessions/{session_id}` | any | Fetch session status. |
| GET | `/v1/managed_agents/sessions/{session_id}/events` | any | SSE stream of harness events. |
| POST | `/v1/managed_agents/sessions/{session_id}/message` | any | Send a follow-up `{"text": "..."}` (or `parts`). |
| DELETE | `/v1/managed_agents/sessions/{session_id}` | any | Stop the Fargate task and mark the session dead. |

## Background reconciler

When `enabled: true`, the proxy starts a background loop (`reconcile_interval_seconds`, default 60s) that:

- Stops Fargate tasks whose session row was deleted in the DB.
- Marks sessions stuck in `creating` for too long as `failed`.
- Tags every task with `litellm_session_id` + `litellm_agent_id` so reconciliation is robust against orphans from crashed proxy processes.

## Gotchas

- **Don't double up the dockerfile path.** The path is resolved relative to the proxy's CWD. If the proxy is launched from `litellm/proxy/`, write `./managed_agents_endpoints/...`, not `./litellm/proxy/managed_agents_endpoints/...`. Or just omit `path` and rely on the bundled-harness lookup.
- **Session creation can take 1–3 minutes the first time.** Cold Fargate task + ECR pull + git clone + harness boot. The proxy waits up to 600s. If it consistently times out, check CloudWatch (`/ecs/litellm-agents` log group) for the offending task.
- **Default VPC required.** If your AWS account has no default VPC in the target region, set `aws.subnets` + `aws.security_group` explicitly.
- **AWS credentials.** boto3's default credential chain is used. Static IAM keys, SSO, instance profiles, all work as long as boto3 can resolve them when the proxy starts.
