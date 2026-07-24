# Public relay V1

The public relay is disabled by default. Deploy the additive Prisma migration, deploy the application with
`PUBLIC_RELAY_ENABLED=false`, publish at least one price through the relay admin page, verify Stripe test mode, and
only then enable registration.

Required runtime settings:

| Variable | Purpose |
| --- | --- |
| `PUBLIC_RELAY_ENABLED` | Master switch |
| `PUBLIC_RELAY_SESSION_SECRET` | URL-safe base64 secret of at least 32 bytes |
| `PUBLIC_RELAY_CONTENT_ENCRYPTION_KEY` | URL-safe base64 AES key of 16, 24, or 32 bytes |
| `PUBLIC_RELAY_CONTENT_ENCRYPTION_KEY_VERSION` | Stored key version, default `1` |
| `PUBLIC_RELAY_TURNSTILE_VERIFY_URL` | Managed Turnstile verification service URL |
| `NEXT_PUBLIC_TURNSTILE_SITE_KEY` | Turnstile widget site key used at dashboard build time |
| `STRIPE_SECRET_KEY` | Stripe server key |
| `STRIPE_WEBHOOK_SECRET` | Stripe endpoint signing secret |
| `PUBLIC_RELAY_CHECKOUT_SUCCESS_URL` | Hosted Checkout success return URL |
| `PUBLIC_RELAY_CHECKOUT_CANCEL_URL` | Hosted Checkout cancellation return URL |
| `RESEND_API_KEY` and `RESEND_FROM_EMAIL` | Production email delivery |
| `SMTP_HOST` and `SMTP_SENDER_EMAIL` | SMTP fallback for development and self-hosting |

Optional settings:

| Variable | Default |
| --- | --- |
| `PUBLIC_RELAY_SESSION_TTL_SECONDS` | `604800` |
| `PUBLIC_RELAY_VERIFICATION_TTL_SECONDS` | `600` |
| `PUBLIC_RELAY_VERIFICATION_RESEND_SECONDS` | `60` |
| `PUBLIC_RELAY_VERIFICATION_MAX_ATTEMPTS` | `5` |
| `PUBLIC_RELAY_MAX_API_KEYS` | `5` |
| `PUBLIC_RELAY_MIN_CHECKOUT_CENTS` | `500` |
| `PUBLIC_RELAY_MAX_CHECKOUT_CENTS` | `50000` |
| `PUBLIC_RELAY_RESERVATION_TTL_SECONDS` | `1800` |
| `PUBLIC_RELAY_CONTENT_RETENTION_DAYS` | `7` |
| `PUBLIC_RELAY_METADATA_RETENTION_DAYS` | `90` |

The proxy requires shared PostgreSQL and Redis. The Stripe webhook endpoint is
`/v1/public/payments/stripe/webhook`. Public keys are restricted to models, chat completions, responses, and
embeddings, and can only see models in the reserved `public-relay-models` access group.
