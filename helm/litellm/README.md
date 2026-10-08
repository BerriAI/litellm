# Custom URL prefix

Set `serverRootPath` to serve the componentized chart under a URL prefix. An empty value or `/` keeps the existing root deployment. A trailing slash is normalized

```yaml
fullnameOverride: litellm
serverRootPath: /services/llm
ingress:
  enabled: false
gateway:
  extraEnv:
    - name: PROXY_BASE_URL
      value: https://platform.example.com/services/llm
backend:
  extraEnv:
    - name: PROXY_BASE_URL
      value: https://platform.example.com/services/llm
```

Supply the chart's existing database, Redis, master-key and SSO settings as usual. Configure the OAuth callback as `https://platform.example.com/services/llm/sso/callback`

The chart passes `SERVER_ROOT_PATH` to gateway, backend and UI. The UI image prepares its export in a fresh directory under `/tmp` at startup, leaving the packaged files untouched. With a read-only root filesystem, mount a writable `emptyDir` on `/tmp`. The default root deployment does not copy or rewrite the export

When the chart's ingress is enabled, its built-in routes and `ingress.extraPaths` are prefixed. Paths in `extraPaths` remain root-relative, for example `/custom-provider`, and the chart adds the configured prefix. The ingress must preserve the path sent upstream. Prefixes containing dots require Gateway API or the ALB controller; ingress-nginx cannot preserve all of the chart's exact matches with its dotted-path normalization

If Lens is enabled, its shared-ingress upload address also includes the prefix. A Lens worker image built with this implementation accepts that prefixed ingestion route while keeping container health probes and internal routes unchanged. An explicit `lensWorker.publicUrl` or dedicated Lens ingress keeps its own address

With an external Gateway API controller, route UI requests to port 3000, inference requests to port 4000, and management/configuration/SSO to port 4001. Preserve the prefix for Python services. The UI accepts both a preserved prefix and a prefix stripped by `URLRewrite`

This Istio example covers the UI, configuration/SSO, chat completions and model listing. Add other inference routes from `templates/ingress.yaml` as needed. Do not send all `/v1` paths to the gateway: several management endpoints also use that version prefix

```yaml
apiVersion: gateway.networking.k8s.io/v1
kind: HTTPRoute
metadata:
  name: litellm
  namespace: litellm
spec:
  parentRefs:
    - name: platform
      namespace: istio-ingress
  hostnames:
    - platform.example.com
  rules:
    - matches:
        - path: {type: Exact, value: /services/llm}
        - path: {type: Exact, value: /services/llm/}
        - path: {type: Exact, value: /services/llm/favicon.ico}
        - path: {type: PathPrefix, value: /services/llm/ui}
        - path: {type: PathPrefix, value: /services/llm/_next}
        - path: {type: RegularExpression, value: '^/services/llm/.*\.txt$'}
      backendRefs:
        - name: litellm-ui
          port: 3000
    - matches:
        - path: {type: PathPrefix, value: /services/llm/v1/chat}
        - path: {type: PathPrefix, value: /services/llm/chat}
        - path: {type: PathPrefix, value: /services/llm/v1/models}
      backendRefs:
        - name: litellm-gateway
          port: 4000
    - matches:
        - path: {type: PathPrefix, value: /services/llm}
      backendRefs:
        - name: litellm-backend
          port: 4001
```

Health probes keep their existing unprefixed paths on the container port. Runtime prefix preparation changes static asset URLs and configuration discovery, and adds HTML metadata so the UI knows its prefix before hydration; login return URLs and dashboard navigation use the prefix reported by the backend
