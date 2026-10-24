# LiteLLM componentized chart

This chart deploys separate gateway, backend and UI services. Lens is off by default. To connect one, deploy Lens from [its own chart](https://github.com/BerriAI/lens/tree/main/helm/lens) and set `lens.enabled: true` with `lens.url` (the address the gateway calls), `lens.publicUrl` (the address agents upload traces to) and the two Secrets Lens created, `lens.gatewaySecret.name` and `lens.serviceTokenSecret.name`, whose keys default to `gateway-secret` and `service-token`

When that Lens runs in the same namespace as this release (an Ingress backend cannot reach a Service in another namespace), `lens.service.name` (and `lens.service.port`, default 4318) adds a `/lens-ingest` route for it to the chart ingress, so agents can upload through the gateway hostname

Follow the [Lens Helm connection guide](https://github.com/BerriAI/lens/blob/main/helm/lens/README.md#connect-a-gateway) for the complete values and verification flow. It also links the GitOps credential requirements and existing-data migration. Source-chart installation requires a built Lens image until the first signed Lens release is published

For an ingress with separate component hostnames, set all three `ingress.componentHosts` values, leave `ingress.host` and `ingress.extraPaths` empty, configure `ui.backendUrl` as the backend's public origin and `backend.gatewayUrl` as the gateway's public origin, and set `ui.publicUrl` to the browser-facing UI origin. The UI Deployment receives `LITELLM_BACKEND_URL` from `ui.backendUrl` only when `ingress.componentHosts` is configured.

```yaml
ingress:
  enabled: true
  host: ""
  componentHosts:
    ui: ui.example.com
    backend: backend.example.com
    gateway: gateway.example.com
  extraPaths: []
ui:
  publicUrl: https://ui.example.com
  backendUrl: https://backend.example.com
backend:
  gatewayUrl: https://gateway.example.com
```
