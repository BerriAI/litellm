# LiteAdmin runtime

LiteAdmin uses the LiteAgents V2 API, released as 0.2.0, with its Pydantic AI harness to run the dashboard's agent on the gateway host. The dashboard sends chat and approval decisions over `/liteadmin/chat`. A separate Python process runs each turn and calls the existing gateway APIs with the signed-in administrator's credential

The runtime has its own environment because the supported LiteAgents harness requires OpenAI 3.x and Anthropic 1.x, while the gateway supports different major versions. Its lockfile pins the SDK's GitHub release wheel and hash. The PyPI package named `liteagents` is unrelated

## Install from source

From the repository root, with Python 3.13 and uv installed:

```sh
UV_PROJECT_ENVIRONMENT="$PWD/.liteadmin-venv" uv sync --project liteadmin --frozen --no-default-groups
export LITEADMIN_PYTHON="$PWD/.liteadmin-venv/bin/python"
export LITEADMIN_RUNTIME="$PWD/liteadmin"
```

The standard Docker images install this environment and set these variables. Other installations must install the runtime alongside the gateway before using LiteAdmin. The runtime requires Python 3.11 or newer; the gateway's base SDK remains compatible with Python 3.10

Reverse proxies must allow WebSocket upgrades on `/liteadmin/chat`. Credentials travel in the first WebSocket message, never in the URL. The browser still asks for consent when the configured inference gateway has a different origin

Management calls default to HTTP on the gateway's loopback port. Set `LITEADMIN_GATEWAY_URL` to a trusted internal gateway URL for TLS-only listeners, mounted path prefixes, or deployments without a loopback TCP listener

## Execution and permissions

Only gateway administrators can start a turn. Tool calls use the existing authenticated management endpoints, so changes to keys, roles, and access still apply. The model only receives the 29 named admin tools, with validated arguments and bounded, redacted results. It has no shell or file tools

Every write presents its exact arguments in the existing approval card. Approving sends only the action ID and decision. The backend executes the arguments it originally proposed. Generated keys go directly to the action card and are removed from model results

Each turn allows six model requests and twelve tool calls. Model and management clients disable retries, and an uncertain write prevents further operations during that turn. Closing or cancelling a turn terminates its process. A write already sent to the gateway may still complete, so the UI marks a lost response as unknown and tells the administrator to check the resource

The runtime is temporary, with a ten-minute limit per turn. Conversation messages and action receipts are sent with each turn; no server conversation database or Temporal service is required. Each connection owns its worker, which also works when the gateway has several server processes

## Tests

```sh
UV_PROJECT_ENVIRONMENT="$PWD/.liteadmin-venv" uv sync --project liteadmin --frozen
PYDANTIC_AI_NO_BANNER=1 .liteadmin-venv/bin/python -m pytest -c liteadmin/pyproject.toml liteadmin/tests
```

The runtime tests use the real LiteAgents and Pydantic AI loops with scripted HTTP transports. They cover lookup results, approval before execution, cancellation, generated-key redaction, uncertain writes, input validation, and result limits
