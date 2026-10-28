# LiteLLM componentized chart

This chart deploys separate gateway, backend and UI services. Configure Lens through `lensWorker.mode`: `disabled` keeps it off, `bundled` installs the Lens dependency, and `external` connects an existing Lens service

Bundled mode generates separate service and identity-signing credentials. External mode requires `lensWorker.gateway.secretName` and `lensWorker.serviceTokenSecret.name`; their keys default to `gateway-secret` and `service-token`. Provision distinct values matching the external Lens deployment

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

## Routes to another Service

For a shared hostname, `ingress.extraPaths` can route to a Service in the same namespace using `serviceName` and `servicePort`. The port can be a number or a named Service port. Omit `service` when using these fields

```yaml
ingress:
  enabled: true
  className: alb
  host: gateway.example.com
  extraPaths:
    - path: /moyai
      pathType: Prefix
      serviceName: moyai-ingress
      servicePort: 8080
```

The Service and application must already support the path prefix. The chart adds the path to the existing Ingress and leaves its built-in routes in place. It does not create the target Service or rewrite requests

For AWS ALB, configure the target Service's `alb.ingress.kubernetes.io/healthcheck-path` annotation when the application needs a different health check from LiteLLM. Existing `service` aliases (`gateway`, `backend`, `ui`) remain supported
