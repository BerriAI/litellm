# LiteLLM componentized chart

This chart deploys separate gateway, backend and UI services. Configure Lens through `lensWorker.mode`: `disabled` keeps it off, `bundled` installs the Lens dependency, and `external` connects an existing Lens service

Bundled mode generates separate service and identity-signing credentials. External mode requires `lensWorker.gateway.secretName` and `lensWorker.serviceTokenSecret.name`; their keys default to `gateway-secret` and `service-token`. Provision distinct values matching the external Lens deployment

Follow the [Lens Helm connection guide](https://github.com/BerriAI/lens/blob/main/helm/lens/README.md#connect-a-gateway) for the complete values and verification flow. It also links the GitOps credential requirements and existing-data migration. Source-chart installation requires a built Lens image until the first signed Lens release is published
