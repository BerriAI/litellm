# LiteLLM componentized chart

This chart deploys separate gateway, backend and UI services

This chart does not deploy Lens. Deploy Lens from [its own chart](https://github.com/BerriAI/lens/tree/main/helm/lens), then set `lens.enabled: true`, `lens.url`, `lens.publicUrl`, `lens.gatewaySecret.name` and `lens.serviceTokenSecret.name`. The secret keys default to `gateway-secret` and `service-token`. Set `lens.service.name` when that Lens runs in this release's namespace and the chart ingress should route `/lens-ingest` to it. The old `lensWorker` values are no longer read, and enabling them fails the render

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
