# Enterprise relay V1

The public relay is disabled by default. Deploy the additive Prisma migration, deploy the application with
`PUBLIC_RELAY_ENABLED=false`, publish at least one price through the relay admin page, create a test enterprise, and
only then enable the relay.

Required runtime settings:

| Variable | Purpose |
| --- | --- |
| `PUBLIC_RELAY_ENABLED` | Master switch |
| `PUBLIC_RELAY_BASE_URL` | Public HTTPS base URL used for one-time links |
| `PUBLIC_RELAY_SESSION_SECRET` | URL-safe base64 secret of at least 32 bytes |
| `PUBLIC_RELAY_CONTENT_ENCRYPTION_KEY` | URL-safe base64 AES key of 16, 24, or 32 bytes |
| `PUBLIC_RELAY_CONTENT_ENCRYPTION_KEY_VERSION` | Stored key version, default `1` |

Optional settings:

| Variable | Default |
| --- | --- |
| `PUBLIC_RELAY_SESSION_TTL_SECONDS` | `604800` |
| `PUBLIC_RELAY_MAX_API_KEYS` | `5` |
| `PUBLIC_RELAY_RESERVATION_TTL_SECONDS` | `1800` |
| `PUBLIC_RELAY_CONTENT_RETENTION_DAYS` | `7` |
| `PUBLIC_RELAY_METADATA_RETENTION_DAYS` | `90` |

The relay requires PostgreSQL and runs with one Uvicorn worker. Sessions, one-time activation and reset tokens, and
rate limits are stored in PostgreSQL. API keys are restricted to models, chat completions, responses, and embeddings,
and can only see models in the reserved `public-relay-models` access group.
