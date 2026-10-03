# LiteAdmin in Slack

Run LiteAdmin alongside your Enterprise gateway using the same Docker image. Admins connect through the gateway's existing login and SSO provider. The Slack worker has no public ingress or separate login domain

The worker runs separately from gateway replicas. Keep one worker per Slack app and retain its encrypted state volume and `CREDENTIAL_ENCRYPTION_KEY` across upgrades

## Requirements

Use a native-enabled LiteLLM image, a valid Enterprise license, a database, an existing HTTPS gateway address, and a gateway model that supports tool calling. Configure SSO on the gateway as usual. Each person connecting must have an active `proxy_admin` account with the same email as their Slack profile

Create and install a Slack app using the [LiteAdmin manifest](https://github.com/BerriAI/litellm-admin-agent/blob/main/slack-manifest.json). Enable Socket Mode, create an app-level token with `connections:write`, and retain the installed bot token and workspace ID. The worker needs outbound Slack connections and HTTPS access to the gateway. Slack does not need an inbound event webhook

## Docker Compose

Build an image containing this integration, then use the LiteAdmin overlay with the normal gateway Compose file

```sh
docker build -t litellm-native-admin:local .
export LITELLM_IMAGE=litellm-native-admin:local
docker compose -f docker-compose.yml -f docker-compose.liteadmin.yml up -d
```

Add these values to your private deployment environment or `.env` file, alongside the gateway's normal license, database and SSO configuration

```dotenv
LITELLM_IMAGE=litellm-native-admin:local
LITELLM_PUBLIC_URL=https://gateway.example.com
LITELLM_ADMIN_MODEL=your-tool-capable-model
SLACK_BOT_TOKEN=your-installed-bot-token
SLACK_APP_TOKEN=your-socket-mode-app-token
SLACK_WORKSPACE_ID=your-workspace-id
ADMIN_AGENT_SERVICE_TOKEN=your-random-shared-secret-at-least-32-characters
CREDENTIAL_ENCRYPTION_KEY=your-persistent-fernet-key
```

Use `openssl rand -hex 32` to generate the shared service token. Generate the encryption key with `python -c 'from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())'`. Store both in your secret manager and keep them out of source control

The overlay uses `--admin-agent` to start the worker from the same image. It publishes no worker port and gives the worker only its own configuration. The gateway uses the private worker address automatically. Keep your existing HTTPS reverse proxy in front of the gateway

## Helm

Enable `liteadmin` in `helm/litellm-helm`. Use an image containing this integration for the chart's normal `image` setting

```yaml
liteadmin:
  enabled: true
  gatewayUrl: https://gateway.example.com
  model: your-tool-capable-model
  existingSecret: liteadmin-slack
  storageSize: 1Gi
  readOnly: false
```

Create `liteadmin-slack` through your usual secret-management workflow with `SLACK_BOT_TOKEN`, `SLACK_APP_TOKEN`, `SLACK_WORKSPACE_ID`, `ADMIN_AGENT_SERVICE_TOKEN`, and `CREDENTIAL_ENCRYPTION_KEY`. Configure the Enterprise license on the gateway using its existing environment settings

The chart creates one worker Deployment with a Recreate strategy, a private ClusterIP Service, and a persistent volume claim. Gateway replicas only receive the service token from this Secret. The worker receives neither the gateway master key nor database credentials. Use `storageClassName` when your cluster needs a particular storage class

Gateway autoscaling does not scale the worker. Do not expose the worker Service through an Ingress or LoadBalancer. Your gateway must be able to reach its private Service, and the worker must be able to reach `gatewayUrl` over HTTPS

## Connect and use

1. Open LiteAdmin in Slack and send `connect`
2. Open the private connection link on your existing gateway
3. Sign in through your normal LiteLLM login and select **Connect account**
4. Return to Slack and ask **List my teams and their current budgets**

LiteLLM checks your current admin role and matching Slack email. The worker verifies both again before accepting the connection, then checks them before agent operations. Connection links expire after ten minutes and can be used once

The personal session lasts 24 hours. Send `connect` to sign in again, or `disconnect` to delete the worker's saved session and invalidate pending links. Disconnect does not revoke an exported credential at the gateway; the credential has its own bounded expiration. Do not copy session credentials out of the worker

Set `ADMIN_READ_ONLY=true` in Compose, or `liteadmin.readOnly: true` in Helm, to restrict the agent to lookups

## Verification and troubleshooting

An unlicensed gateway returns 403 on the connection page. A non-admin or an account whose email differs from Slack also receives 403. An expired, disconnected or previously consumed link returns 410

The gateway requires `LITELLM_ADMIN_AGENT_URL` and `ADMIN_AGENT_SERVICE_TOKEN`; the deployment examples configure them automatically. The worker uses `CONNECTION_AUTH_MODE=native`, set by `--admin-agent`. A 503 on the connection page usually means the private worker is unavailable, the shared service tokens differ, or current permissions could not be checked

This mode does not use `/register`, `/authorize`, `/token`, a hosted callback allowlist, or a second identity-provider client. Existing standalone agent installations can continue using their own configured login mode
