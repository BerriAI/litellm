# Develop the LiteLLM integration with Lens

This contributor guide covers the boundary between LiteLLM and [Lens](https://github.com/BerriAI/lens). Lens owns its UI, tracing, investigations, signals, datasets, evaluations and ClickHouse storage. LiteLLM embeds the shared UI package, delegates authenticated requests, and forwards gateway telemetry

For product setup, use Lens's [standalone quickstart](https://github.com/BerriAI/lens/blob/main/deploy/lens/README.md), [Helm guide](https://github.com/BerriAI/lens/blob/main/helm/lens/README.md), [analysis guide](https://github.com/BerriAI/lens/blob/main/docs/analysis.md), or [agent-assisted setup](https://github.com/BerriAI/lens/blob/main/docs/setup-with-agent.md). Keep those instructions in the Lens repository

This companion change is under qualification. The vendored UI and chart are development artifacts, not a published release. Existing Lens data needs transfer before the clean cutover; follow the [Lens migration guide](https://github.com/BerriAI/lens/blob/main/docs/migration.md) and its qualification requirements

## Connect through either LiteLLM Helm chart

For `lensWorker.mode=bundled`, both charts generate separate service and identity-signing credentials and share the matching references with Lens. For `lensWorker.mode=external`, supply both references to the credentials already configured in Lens:

```yaml
lensWorker:
  mode: external
  externalUrl: http://lens.lens.svc.cluster.local:4318
  publicUrl: https://lens.example.com
  gateway:
    secretName: lens-connection
    secretKey: gateway-secret
  serviceTokenSecret:
    name: lens-connection
    key: service-token
```

Replace the service and public addresses with those reachable from your gateway and agents respectively. Provision `lens-connection` in the gateway's namespace through your secret manager. Its two keys must contain the separate values described below, matching the Lens deployment. The identity-signing key grants delegated user authority; do not reuse the service-token value for it

For GitOps, follow the Lens chart's [offline rendering instructions](https://github.com/BerriAI/lens/blob/main/helm/lens/README.md#render-with-gitops). Generated credentials depend on live Helm lookup and must be provisioned separately for offline renderers

## Connect a development gateway

Start Lens using its quickstart, then run the gateway directly on the same host with its existing configuration and database settings. The optional [gateway Compose example](../docker/docker-compose.tracing.yml) connects to an existing Lens deployment; its PostgreSQL service belongs to LiteLLM

Generate two separate private values with `openssl rand -hex 32`. Add them to the Lens environment file, `deploy/lens/.env` in the Lens checkout:

```dotenv
LITELLM_LENS_SERVICE_TOKEN=<first-generated-value>
LENS_GATEWAY_SECRET=<second-generated-value>
```

Supply the same values to the gateway process, along with its Lens addresses:

```dotenv
LITELLM_LENS_URL=http://127.0.0.1:4318
LITELLM_LENS_PUBLIC_URL=http://localhost:4318
LITELLM_LENS_SERVICE_TOKEN=<first-generated-value>
LENS_GATEWAY_SECRET=<second-generated-value>
```

`LITELLM_LENS_URL` is the base address the gateway can reach. `LITELLM_LENS_PUBLIC_URL` is the base address shown to browsers and agent exporters. A gateway running in a container needs a container-reachable address, not its own `127.0.0.1`. Both addresses point at Lens, without `/v1/traces` or `/lens` appended

The service token authorizes the gateway's internal trace reads, spend records and feedback delivery. The signing secret authenticates short-lived identities forwarded by the gateway. Neither is a user's API key or a tracing key. Keep them in private environment files or your deployment's secret manager

Apply the environment change using the Lens quickstart's configuration instructions, then restart the gateway. Check the connection using an existing gateway API key:

```sh
curl --fail http://localhost:4000/lens/service \
  -H "Authorization: Bearer $LITELLM_API_KEY"
```

Expect `connected: true`, `status.storage_ready: true` and `status.public_contract: 1`. `status.protocol_version` is the internal worker protocol and can differ from the public contract. If the connection is unavailable, check both secrets, the gateway-to-Lens URL, and Lens readiness before retrying

Sign into LiteLLM and open `/ui/lens/`. Follow the Lens quickstart's first-trace flow inside the embedded page, then inspect that exact run under **Traces**. Readiness alone does not verify ingestion, trace visibility or provider access

## Preserve the gateway boundary

The embedded page uses the LiteLLM session without a second Lens login. The gateway derives delegated identities from its authenticated user and trace-read scope. Administrators retain administrator access, viewers remain read-only, and ordinary users keep their existing user and permitted-team trace scope. Server secrets must stay out of browser props, assets and responses

The adapter forwards Lens requests under `/lens` using the supported public contract. Lens implements the product behavior and owns its records. Keep ordinary inference authentication, model routing, spend accounting and gateway-owned storage unchanged when editing this integration

Gateway telemetry delivery uses a bounded asynchronous queue and bounded retries. A Lens outage must not block ordinary inference; delivery can drop records after its retry or queue limit. Validate both the successful delivery path and the unavailable-service path when changing the relay

The existing `clickhouse` callback remains an independent gateway spend writer. It uses `CLICKHOUSE_URL`, `CLICKHOUSE_DATABASE` and `AGENT_TRACING_RETENTION_DAYS`, and owns only `spend_logs` and its `_litellm_spend_migrations` ledger. Its existing schema-provisioning precondition is preserved: `ClickHouseBatchLogger` does not create tables automatically. The [native isolation qualification](evidence/lens-native-isolation.json) explicitly calls `await logger.storage.ensure_schema()` before exercising a fresh database. Lens provisions its own schema during startup separately

## Route analysis through the development gateway

Skip this section if Lens connects directly to an analysis provider. The Lens analysis guide owns provider configuration; the gateway-specific settings below enable signed analysis markers and preserve billing

To route analysis through LiteLLM, create a normal gateway model key with access to your analysis model and supply it to Lens. Keep its existing budgets and billing settings. Add the explicit gateway base URL so Lens can mark its own inference requests:

```dotenv
LENS_GATEWAY_URL=https://gateway.example.com/v1
LENS_GATEWAY_MODEL_KEY=<gateway-model-key>
LENS_ANALYSIS_MODELS='[{"name":"analysis","model":"openai/gpt-6.1-sol","provider":"openai_compatible","api_base":"https://gateway.example.com/v1","api_key_env":"LENS_GATEWAY_MODEL_KEY","output_limits":{"max_tokens":4096}}]'
```

Replace the hostname and model with the gateway deployment's actual values. Keep `LENS_GATEWAY_SECRET` identical on both services, then restart Lens. See [analysis model configuration](https://github.com/BerriAI/lens/blob/main/docs/analysis.md) for provider, price and capacity settings

Lens attaches a short-lived signed marker only to requests under that explicit gateway origin and base path. The gateway still authenticates the model key and records billing, while keeping the analysis prompts out of message logs and recursive Lens analysis. Direct-provider requests do not carry the marker

## Update the shared UI

Lens changes belong in its repository. LiteLLM contains the thin gateway adapter and embedded wrapper; it consumes the same Lens UI package used by the standalone product. Run the Lens quickstart for backend and UI development instead of the retired `make lens-dev` helper

The LiteLLM dashboard uses the checked-in `@litellm/lens-ui` tarball and npm lockfile. This branch pins its UI to [Lens commit `5846697965ba06a0370f83220b12d217884df9da`](https://github.com/BerriAI/lens/commit/5846697965ba06a0370f83220b12d217884df9da). All 389 packaged files match that commit; [qualification evidence](evidence/lens-latest-ui.json) records the package checksum, npm integrity and source comparison. Update the package, lockfile and provenance together when adopting a new UI release

From `ui/litellm-dashboard`, run `npm ci` followed by `npm run dev` to work on gateway integration. The focused packaged-UI check is:

```sh
npx vitest run src/components/lens/EmbeddedLens.integration.test.tsx \
  'src/app/(dashboard)/lens/page.test.tsx'
```

A source installation with `pip install .` does not build the dashboard. To serve the current UI from that installation, run `npm ci` and `bash build_ui.sh` from `ui/litellm-dashboard` before packaging or starting the gateway. This refreshes the local `litellm/proxy/_experimental/out` directory. Official image and wheel builds receive the compiled UI from the release pipeline

For gateway adapter and telemetry changes, run the focused Python checks from the repository root:

```sh
uv run pytest -q tests/unit/proxy/lens tests/unit/tracing \
  tests/unit/proxy/test_tracing_endpoints.py
```

## Qualify independent releases

Gateway releases consume a qualified Lens release's UI, chart and image digest. The public UI/API contract is versioned separately from the internal worker protocol. A Lens runtime update does not rebuild or roll out the gateway; changing the embedded UI package requires a gateway UI release

Keep the existing chart family, gateway service names, selectors, persistent volumes and credential secrets when adopting the shared Lens chart. The installation smoke workflow checks separately pinned Lens versions, gateway-only and Lens-only updates, and retained data after rollback. Its live Kubernetes run is a release qualification step, not a substitute for transferring an existing installation's Lens data

When PostgreSQL and the migration Job's dependencies already exist, select the [Helm migration hook settings](../helm/litellm-helm/README.md#migration-job-settings) for Helm CLI deployments. Keep one migration controller: Helm hooks for Helm installations or ArgoCD hooks for Argo-managed installations. Neither Helm `pre-install` nor ArgoCD `PreSync` hooks can use bundled PostgreSQL created in the same install or sync. The installation smoke workflow provisions PostgreSQL before installing either gateway chart

The installation workflow requires a `gateway_baseline_ref` containing the full commit of a previously qualified gateway integration. It builds baseline gateway, backend and monolith images separately, then upgrades those images to the selected workflow commit while keeping the Lens pod UID, image ID and restart count unchanged. The split chart keeps its current UI and migration images throughout this comparison. Qualification applies to the selected source pair; it does not promise compatibility with every earlier gateway release
