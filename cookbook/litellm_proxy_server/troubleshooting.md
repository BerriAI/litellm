# LiteLLM Proxy Config Troubleshooting

Quick fixes for common proxy setup issues. For full reference see https://docs.litellm.ai/docs/simple_proxy.

## Proxy won't start

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| `Config file not found` | Wrong path to `config.yaml` | Pass `--config /path/to/config.yaml` or set `CONFIG_FILE` env var. |
| `Database connection failed` | `DATABASE_URL` missing or invalid | Set `DATABASE_URL` for features requiring Prisma (keys, MCP DB, spend logs). |
| `Port already in use` | Another process on 4000 | Change port: `litellm --port 4001` or stop the conflicting process. |
| YAML parse error | Indentation or invalid key | Validate YAML; check keys against `proxy_server_config.yaml` example in the repo root. |

## Model returns 401 / authentication errors

```yaml
# Correct: read API key from environment
litellm_params:
  api_key: os.environ/OPENAI_API_KEY

# Wrong: literal string (not interpolated)
litellm_params:
  api_key: OPENAI_API_KEY
```

Verify the env var is set in the shell or container:

```bash
echo $OPENAI_API_KEY   # Linux/macOS
echo $env:OPENAI_API_KEY  # PowerShell
```

## Model not found / routing errors

1. Check `model_name` in your request matches an entry in `model_list`.
2. Confirm `litellm_params.model` uses the correct provider prefix (`openai/`, `anthropic/`, `azure/`, etc.).
3. For router deployments, ensure the model group name is used—not the underlying provider model ID.

## MCP tools not appearing

| Check | Command / action |
|-------|------------------|
| Server registered | `GET /mcp-rest/tools/list` with your API key |
| Key has MCP access | Verify key's `object_permission` includes the server or access group |
| Server reachable | `POST /mcp-rest/test/connection` from the Admin UI or API |
| OAuth not completed | For OAuth2 servers, authorize via Admin UI → MCP → Authorize and Fetch |

### MCP connection errors

| Error message | Fix |
|---------------|-----|
| `server is unreachable` | Check `url`, firewall, and that the upstream MCP server is running. |
| `request header is malformed` | Inspect `static_headers` for trailing spaces or illegal characters. |
| `missing_user_env_vars` (412) | Configure per-user env vars in the MCP server settings UI. |

## Spend / logging not recording

- Set `DATABASE_URL` and run migrations (`litellm-proxy-extras`).
- Enable success/failure callbacks in `litellm_settings` or per-key metadata.
- For local dev without a DB, spend tracking is limited to in-memory caches.

## Debug mode

```bash
export LITELLM_LOG=DEBUG
litellm --config config.yaml
```

Or in config:

```yaml
litellm_settings:
  set_verbose: true
```

## Health checks

```bash
curl http://localhost:4000/health/liveliness
curl http://localhost:4000/health/readiness
```

## Still stuck?

1. Search existing issues: https://github.com/BerriAI/litellm/issues
2. Include: proxy version, redacted config snippet, error log, and request/response (no secrets).
3. Community Discord: https://discord.gg/wuPM9dRgDw