# Use Lens from LiteLLM

Lens owns the tracing UI, investigations, signals, datasets and evaluations in [BerriAI/lens](https://github.com/BerriAI/lens). LiteLLM embeds its UI package and forwards authenticated Lens requests to the Lens service. Lens stores its data in ClickHouse and can run without a LiteLLM gateway

This companion change is under qualification. The vendored UI and chart are development artifacts, not a published release. Do not use this branch as an upgrade procedure for an existing installation until release provenance and migration qualification are complete

## Connect a local gateway

Start Lens using its [standalone quickstart](https://github.com/BerriAI/lens/blob/main/deploy/lens/README.md). For this example, run the gateway directly on the same host. Keep its existing gateway configuration and database settings

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

Apply the Lens environment change from the Lens checkout:

```sh
docker compose -f deploy/lens/compose.yaml up -d --wait
```

Restart the gateway with its new environment. Sign into LiteLLM, open `/ui/lens/`, then open **Settings > Tracing > Connect an agent**. Create a tracing key and send a run directly to the Lens endpoint shown in setup. Open the run under **Traces** and inspect its messages

The embedded page uses your LiteLLM session. It does not require a second Lens login. Gateway administrators retain administrator access; viewers remain read-only, and ordinary users keep the gateway's existing user and permitted-team trace scope. Standalone Lens login continues to work independently

Check the connection using an existing gateway API key:

```sh
curl --fail http://localhost:4000/lens/service \
  -H "Authorization: Bearer $LITELLM_API_KEY"
```

Expect `connected: true`, `status.storage_ready: true` and `status.public_contract: 1`. `status.protocol_version` is the internal worker protocol and can differ from the public contract. If the connection is unavailable, check both secrets, the gateway-to-Lens URL, and Lens readiness before retrying

The optional [gateway Compose example](../docker/docker-compose.tracing.yml) connects to an existing Lens deployment. Its PostgreSQL service belongs to LiteLLM. Lens still requires only ClickHouse

## Run Lens analysis through LiteLLM

Skip this section if Lens connects directly to an analysis provider. Embedding the UI does not change your model routing

To route analysis through LiteLLM, create a normal gateway model key with access to your analysis model and supply it to Lens. Keep its existing budgets and billing settings. Add the explicit gateway base URL so Lens can mark its own inference requests:

```dotenv
LENS_GATEWAY_URL=https://gateway.example.com/v1
LENS_GATEWAY_MODEL_KEY=<gateway-model-key>
LENS_ANALYSIS_MODELS='[{"name":"analysis","model":"openai/gpt-6.1-sol","provider":"openai_compatible","api_base":"https://gateway.example.com/v1","api_key_env":"LENS_GATEWAY_MODEL_KEY","output_limits":{"max_tokens":4096}}]'
```

Replace the hostname and model with the gateway deployment's actual values. Keep `LENS_GATEWAY_SECRET` identical on both services, then restart Lens. See [analysis model configuration](https://github.com/BerriAI/lens/blob/main/docs/analysis.md) for provider, price and capacity settings

Lens attaches a short-lived signed marker only to requests under that explicit gateway origin and base path. The gateway still authenticates the model key and records billing, while keeping the analysis prompts out of message logs and recursive Lens analysis. Direct-provider requests do not carry the marker

## Develop and release independently

Lens changes belong in its repository. LiteLLM contains the thin gateway adapter and embedded wrapper; it consumes the same Lens UI package used by the standalone product. Run the Lens quickstart for backend and UI development instead of the retired `make lens-dev` helper

The LiteLLM dashboard uses the checked-in `@litellm/lens-ui` tarball and npm lockfile. This branch pins its UI to [Lens commit `b0f3d5b447eae749b2665c7fbbaf19c71d4d3e38`](https://github.com/BerriAI/lens/commit/b0f3d5b447eae749b2665c7fbbaf19c71d4d3e38). All 394 packaged files match that commit; [qualification evidence](evidence/lens-gateway.json) records the package checksum, npm integrity and source comparison

From `ui/litellm-dashboard`, run `npm ci` followed by `npm run dev` to work on gateway integration. The focused packaged-UI check is:

```sh
npx vitest run src/components/lens/EmbeddedLens.integration.test.tsx \
  'src/app/(dashboard)/lens/page.test.tsx'
```

Gateway releases consume a qualified Lens release's UI, chart and image digest. The public UI/API contract is versioned separately from the internal worker protocol. A Lens runtime update does not rebuild or roll out the gateway; changing the embedded UI package requires a gateway UI release

Keep the existing chart family, service names, selectors, persistent volumes and credential secrets when adopting the shared Lens chart. The installation smoke workflow checks separately pinned Lens versions, gateway-only and Lens-only updates, and retained data after rollback. Its live Kubernetes run is a release qualification step, not a substitute for checking an existing installation's migration requirements
