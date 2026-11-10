# LiteLLM componentized chart

This chart deploys separate gateway, backend and UI services

The external Lens integration has been removed. Existing `lens` and `lensWorker` values are ignored with a warning in the release notes. Remove these obsolete values when updating your configuration.

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

## Enterprise billing metrics

Billing metrics use Cognito bearer authentication. Create the Secret with
`client-id` and `client-secret` keys, then enable billing metrics:

```bash
kubectl create secret generic litellm-billing-metrics --from-literal=client-id=... --from-literal=client-secret=...
```

```yaml
billingMetrics:
  enabled: true
```

The chart requires an HTTPS endpoint, an HTTPS token URL, and the Secret name
when billing metrics are enabled

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
