# Docker Secret Manager

Read secrets from [Docker secrets](https://docs.docker.com/engine/swarm/secrets/) mounted on the filesystem.

| Feature | Support | Description |
|---------|----------|-------------|
| Reading Secrets | ✅ | Read secrets e.g. `OPENAI_API_KEY` |
| Writing Secrets | ❌ | Docker secrets are managed by the Docker daemon |

## How it works

Docker mounts each secret as a plain file under `/run/secrets/<secret_name>` (configurable). LiteLLM reads the file contents and uses the value as the secret. Trailing whitespace is stripped automatically, so secrets created with `echo "value" | docker secret create ...` work as expected.

Works with both **Docker Swarm** (secrets encrypted at rest and in transit, recommended for production) and **Docker Compose** (bind-mounted files, suitable for local development).

## Setup

**Step 1.** Create your secrets

```bash
# Docker Swarm
echo "sk-..." | docker secret create openai_api_key -
echo "sk-ant-..." | docker secret create anthropic_api_key -

# Docker Compose — write plain files into your secrets directory
echo "sk-..." > /run/secrets/openai_api_key
echo "sk-ant-..." > /run/secrets/anthropic_api_key
```

**Step 2.** Mount secrets in your container

```yaml
# docker-compose.yml
services:
  litellm:
    image: ghcr.io/berriai/litellm:main-latest
    secrets:
      - openai_api_key
      - anthropic_api_key

secrets:
  openai_api_key:
    file: ./secrets/openai_api_key   # path to the secret file on the host
  anthropic_api_key:
    file: ./secrets/anthropic_api_key
```

**Step 3.** Add to your proxy `config.yaml`

```yaml
general_settings:
  key_management_system: "docker"
  key_management_settings:
    secrets_path: "/run/secrets"  # optional — this is the default

model_list:
  - model_name: gpt-4o
    litellm_params:
      model: openai/gpt-4o
      api_key: os.environ/OPENAI_API_KEY   # resolved from env var

  - model_name: claude-sonnet
    litellm_params:
      model: anthropic/claude-sonnet-4-5
      api_key: os.environ/ANTHROPIC_API_KEY
```

**Step 4.** Start the proxy

```bash
litellm --config /path/to/config.yaml
```

## Secret name resolution

LiteLLM resolves secret names using the following fallback order:

1. **Exact match** — looks for `/run/secrets/<name>` as given
2. **Lowercase fallback** — if the exact name isn't found, tries the lowercase version

This means you can use either `OPENAI_API_KEY` or `openai_api_key` in your config and LiteLLM will find the secret file regardless of how it was named on disk.

```bash
# Both of these will be found when you call get_secret("OPENAI_API_KEY")
/run/secrets/OPENAI_API_KEY
/run/secrets/openai_api_key   # ← lowercase fallback
```

## Using `os.environ/` as a fallback

If you want a specific key to fall back to an environment variable when the Docker secret isn't present, use the `os.environ/` prefix:

```yaml
model_list:
  - model_name: gpt-4o
    litellm_params:
      model: openai/gpt-4o
      api_key: os.environ/OPENAI_API_KEY  # reads $OPENAI_API_KEY from env
```

Without the `os.environ/` prefix, a value that isn't found as a Docker secret file is returned as-is (the literal string), so use literal values only when you intend to pass them directly.

## Configuration options

| Setting | Required | Default | Description |
|---------|----------|---------|-------------|
| `key_management_system` | Yes | — | Must be `"docker"` |
| `key_management_settings.secrets_path` | No | `/run/secrets` | Directory where Docker secrets are mounted |
| `key_management_settings.access_mode` | No | `read_only` | Must be `"read_only"` or `"read_and_write"` (write operations are not supported and will raise an error) |

## Security notes

- Secret values are **never logged** — only secret names appear in log output.
- **Path traversal** is blocked: names containing `..` or absolute paths raise an error.
- **Symlinks** that point outside the secrets directory are blocked.
- The manager is **read-only** by design. Use `docker secret create` / `docker secret rm` to manage secrets.
