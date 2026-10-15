# How selected offerings become a client catalog

Automatic supplier metadata is the normal path for AUTO offerings. Selecting an OpenRouter endpoint is a routing decision, not an instruction to replace discovery with handwritten capabilities. Keep overrides for an intentional operator policy or an evidenced missing value, not merely because a model was configured

The schema still requires an explicit `source`. Normal automatic usage selects `source: auto` without copying a capability card; it does not mean an omitted `source` field is accepted

This walkthrough is verified against runtime commit `ec1e681d95dca0e6660e21fba9060ec9e3c7afde`. Relative source links follow the code in the same checkout. The feature extends native LiteLLM adapters, types, authentication, listing and Router paths; referencing those paths does not mean every implementation in them originated in this PR. See the [setup guide](README.md) for configuration and lifecycle instructions

```text
supplier connection -> inventory -> native normalization -> offering resolution
                    -> Router generation -> authorized gateway catalog
                    -> client options -> request mapping -> supplier
```

## 1. Connection, offering and metadata are different layers

The [configuration types](../../litellm/types/proxy/model_offerings.py) separate these concerns

| Layer                         | What it owns                                                                                                                | What it does not imply                                                        |
| ----------------------------- | --------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------- |
| Supplier connection           | Native provider/protocol, trusted or configured API base, authentication and headers                                        | That every model at the supplier becomes public                               |
| Selected offering             | Public `model_name`, `upstream_model`, connection reference, enabled state, AUTO/MANUAL source and optional provider policy | That the operator must write a capability card or implement a provider plugin |
| Metadata and request defaults | Limits, capabilities, effort choices/default, API surfaces/modalities and optional output-budget policy                     | That routing selection, UI choices and physical limits are interchangeable    |

The `provider` on an offering references a connection name. That connection's `provider` selects an existing adapter, currently `chatgpt`, `openrouter`, `vercel_ai_gateway` or generic `openai` in this external-file schema. A custom connection name or public alias does not create custom provider code. A service that already speaks the supported OpenAI-compatible protocol can use the generic adapter; new provider implementation is needed only when its authentication, protocol or request/response behavior actually requires one

## 2. Retrieve an inventory, without exposing it wholesale

The [native inventory dispatcher](../../litellm/llms/model_inventory.py) chooses the supplier path. Discovery happens on the gateway using the selected connection's identity. A client does not need a second connection or credentials for the supplier

| Connection                 | Inventory and normalization path                                                                                                                                                                                                                                                                         |
| -------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Native ChatGPT account     | The [ChatGPT adapter](../../litellm/llms/chatgpt/model_info.py) reads the saved server OAuth account and calls the trusted backend's `/models?client_version=...`. An explicit base must match that backend. Discovery requests existing access tokens with device login disabled                        |
| OpenRouter                 | The [OpenAI-compatible inventory](../../litellm/llms/openai_like/model_info.py) fetches the configured models catalog. Pinned AUTO offerings additionally use the [OpenRouter endpoint adapter](../../litellm/llms/openrouter/model_info.py) for `/models/{author}/{slug}/endpoints` under that API base |
| Vercel AI Gateway          | Uses the compatible catalog request, then interprets Vercel's model kind, modalities, reasoning options and language-model output-budget field                                                                                                                                                           |
| Generic compatible service | Uses the configured `/v1/models` catalog and recognized metadata fields, without guessing a model family from its name                                                                                                                                                                                   |

`add_path_to_api_base` handles an API base that already includes its version path. Discovery does not blindly append another `/v1` to it. Provider-specific native getters resolve OpenRouter/Vercel bases rather than maintaining a second URL registry

Endpoint requests are deduplicated and restricted to enabled, selected AUTO models with a policy and a successful model-inventory membership result. The implementation does not request endpoint details for every model in a large supplier catalog. MANUAL metadata resolution ignores inventory, although the manager may still refresh configured connections

Three decisions remain separate: a supplier reports a model, the operator selects and enables its public offering, and a caller is authorized to discover/use it. Downloading a model does not automatically enable it or grant it to any key

## 3. Normalize meaning, not just field names

[ModelMetadata and ModelRequestDefaults](../../litellm/types/router.py) provide the shared native facet used by inventory, deployments and listing

| Field                                                                                   | Meaning                                                                                                        |
| --------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------- |
| `context_window`                                                                        | Combined input and generated-output token capacity                                                             |
| `max_input_tokens`                                                                      | Independently known input ceiling, not context minus output                                                    |
| `max_output_tokens`                                                                     | Independently known generated-output ceiling, not a requested default                                          |
| `supports_function_calling`, `supports_parallel_function_calling`, `supports_reasoning` | Known capabilities, including an explicit false when the source supports that conclusion                       |
| `reasoning_effort_levels`                                                               | Exact accepted effort choices, not a guessed low/medium/high ladder                                            |
| `default_reasoning_effort`                                                              | A compatible advertised default or normalized API-facing default, subject to the provider qualifications below |
| `supported_endpoints`, `supported_modalities`, `supported_output_modalities`            | Known API surfaces and input/output modalities                                                                 |
| `request_defaults`                                                                      | Optional requested output budgets, separate from all physical limits                                           |

Unknown optional fields are omitted after applicable sources are resolved. False and an empty list remain meaningful values. Numeric zero in supplier limits is treated as unknown; malformed supplier metadata is not evidence of a successful empty inventory. Validation also prevents an effort default outside a known choice list and clears choices/defaults when reasoning is explicitly unsupported

### Provider-specific interpretations

The generic card adapter recognizes native fields and context aliases such as `max_model_len` and `context_length`. For unpinned OpenRouter cards it also reads `top_provider`, architecture and model-level reasoning information. For Vercel, `type` can supply chat/embedding mode, `reasoning_options` supplies effort choices, and `max_tokens` is treated as output capacity only for a language model, not blindly for embeddings

Native ChatGPT's Codex `context_window` and `max_context_window` are retained as `codex_context_window` and `codex_max_context_window` in inventory/deployment metadata. They are not unconditionally promoted to generic physical `context_window` or independent input capacity. Independently reported input/output fields can still be used. These Codex-specific extras are not part of the generic `/v1/models` projection, so their retention is not a promise that every client receives them

ChatGPT discovery removes `ultra` from API effort choices because it is a multi-agent UI choice. It translates the catalog spelling `persistent` to advertised API spelling `disabled`. For a native default of `ultra`, the current normalization chooses a supported `multi_agent_reasoning_effort`, otherwise supported `max`, otherwise the last remaining reported API effort, otherwise omits the default. This is an explicit UI-to-API normalization policy, not proof that multi-agent Ultra is equivalent to a particular single-request effort or that the fallback is a measured backend default

OpenRouter effort choices describe OpenRouter's gateway contract, not an independently verified native enum for a supplier such as StreamLake. Explicit `reasoning.supported_efforts: null` expands to the documented gateway effort set; omission does not declare choices; `[]` remains empty. Mandatory reasoning excludes `none`. A disabled-by-default model does not claim an active default effort, while an explicitly reported compatible `none` default can remain. Boolean reasoning support alone never creates choices

## 4. Resolve AUTO, pinned AUTO and MANUAL

The precedence is implemented by `_deployment` in the [offering manager](../../litellm/proxy/model_offerings.py), not by a client-side table

| Mode                    | Availability                                                              | Metadata resolution, lowest to highest priority                                                                                        | Missing data                                                                                         |
| ----------------------- | ------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------- |
| AUTO, unpinned          | Enabled and present in the connection's latest successful model inventory | Provider-appropriate bundled metadata fallback, then downloaded normalized supplier fields, then explicit non-null `model_info` fields | Missing supplier fields can still have known bundled fallback. Remaining unknown fields are omitted  |
| AUTO, pinned OpenRouter | Same model-membership rule, with endpoint policy retained                 | Metadata from eligible endpoints plus applicable model-level reasoning information, then explicit non-null `model_info` fields         | No unrestricted aggregate or bundled capacity substitution. Unknown endpoint guarantees stay omitted |
| MANUAL                  | Enabled, independent of supplier inventory membership                     | Handwritten `model_info` only                                                                                                          | Missing metadata remains unknown. The existing protocol adapter can still perform inference          |

For unpinned AUTO fallback, the manager looks up the provider-prefixed model in the native loaded cost map and permits an unprefixed fallback for generic `openai`. It reuses the native catalog rather than building another pricing/model registry. Consequently, a field on an AUTO GPT offering need not have come directly from the latest live JSON. Unpinned OpenRouter metadata is a model-level card; it does not certify which endpoint will serve an individual future request

For pinned AUTO, `OpenRouterProviderPolicy.allows_endpoint` matches full slugs and base/variant prefixes, with service-tier suffixes requiring explicit selection. For every independently known limit, the common native aggregator takes the minimum across eligible endpoints. A limit unknown on any eligible endpoint stays unknown. Capability guarantees and lists are combined conservatively, and a default is retained only when compatible across sources. Status, uptime, price and performance do not change this metadata eligibility calculation or operator availability policy

Effort choices/defaults are combined with endpoint metadata only where eligible endpoints declare `reasoning_effort` support. More restrictive endpoint declarations narrow model-level choices. An endpoint permitting reasoning without the effort control does not justify advertising an effort selector. Missing/failed endpoint discovery never borrows a larger context from another supplier

### What an override actually does

An explicit valid, non-null field overrides that field, not the entire downloaded card. A false capability or `[]` choice list survives the merge, subject to native consistency validation. `null` is omitted by the configured metadata dump and therefore does not clear an inherited value. A supplied `request_defaults` object replaces that field as an object; this is not a recursive merge of its budget map

The override facility is optional. Copied endpoint values used as a temporary workaround are not automatically an intentional long-term policy. Remove unnecessary copied limit/capability fields from an AUTO definition to let discovery supply and refresh them; retain genuine operator policy and required configuration such as pinned `mode: chat`. Keeping the facility does not require keeping a workaround forever

For a sole eligible endpoint reporting context `300000` and output `30000`, AUTO without intentional overrides advertises those bounds. If it later reports `280000` and `24000`, the next successful refresh publishes those values for subsequent requests. An unknown independent input ceiling remains omitted in both generations. These are illustrative numbers, not a claim about a named supplier

## 5. Configure selection without copying a card

These snippets illustrate standalone external-file definitions. Replace placeholder model IDs with exact IDs from the relevant catalog. They do not assert current capabilities for a vendor model or modify an existing deployment

### Native GPT offering, automatically described

```yaml
version: 1
providers:
  account:
    provider: chatgpt
offerings:
  - model_name: gpt-auto
    provider: account
    upstream_model: catalog-model-slug
    source: auto
    enabled: true
```

No capability block is needed to enable discovery. The account must already be authorized through the native server mechanism. Add a metadata field only for an intentional exception or evidenced information the automatic sources do not provide

### Endpoint selection, still automatically described

```yaml
version: 1
providers:
  aggregator:
    provider: openrouter
    api_key: os.environ/OPENROUTER_API_KEY
offerings:
  - model_name: pinned-auto
    provider: aggregator
    upstream_model: organization/model
    source: auto
    enabled: true
    provider_policy:
      provider: openrouter
      only: [supplier/fp8]
      allow_fallbacks: false
    model_info:
      mode: chat
```

The current schema requires the matching OpenRouter supplier and explicit chat mode for a policy. It does not require static context, output or reasoning fields. Routing selection and automatic metadata remain separate

### Deliberately maintained compatible offering

```yaml
version: 1
providers:
  custom:
    provider: openai
    api_base: https://supplier.example/v1
    api_key: os.environ/CUSTOM_SUPPLIER_API_KEY
offerings:
  - model_name: custom-flash
    provider: custom
    upstream_model: organization/model
    source: manual
    model_info:
      mode: chat
      context_window: 262144
      supports_function_calling: true
      supports_parallel_function_calling: true
      supports_reasoning: true
      reasoning_effort_levels: [low, high, xhigh, max]
      supported_endpoints: [/v1/chat/completions]
      supported_modalities: [text, image]
      supported_output_modalities: [text]
      request_defaults:
        output_token_budget: 65536
        output_token_budget_by_reasoning_effort:
          max: 131072
```

This is an explicit MANUAL use case, such as a compatible service without a usable inventory. The context and capability block are operator assertions requiring evidence, not supplier-discovered proof. There is deliberately no independent `max_output_tokens` capability. The output budgets do not establish one. Naming a connection `custom` or using this alias still invokes the generic `openai` adapter, not a custom model-family implementation

## 6. A requested budget is not a capability

The [budget helper](../../litellm/router_utils/model_request_defaults.py) defaults requests only for supported chat/Responses/Messages call types when a `request_defaults` policy exists and the caller has not supplied a non-null `max_tokens`, `max_completion_tokens` or `max_output_tokens`

It chooses the budget's effort key from request effort first, then deployment request settings, then advertised `default_reasoning_effort`. It understands a direct effort string/object and `reasoning.effort`. An exact matching effort budget wins over the base output budget. If neither supplies a budget, nothing is injected. There is no guessed effort-strength conversion

The helper returns `max_output_tokens` for Responses calls and `max_tokens` for the other supported calls. It does not inject a reasoning effort just because a default was advertised. An explicit caller budget wins over this configured default-budget policy, and conflicting deployment budget parameters are removed in that policy path. Native/provider request translation still determines the eventual wire format

For the manual example, an omitted effort uses the base requested budget of `65536`; explicit `max` selects `131072`. Neither value means the physical output ceiling is known. This helper is defaulting, not a provider-independent quota or proof that a chosen budget is feasible

An advertised reasoning default, a client electing to send that default, an explicit deployment request setting and a provider adapter's own default behavior are different things. The offering manager itself does not turn every advertised default into an outgoing request parameter. With no explicit effort setting or provider-specific injection, omission leaves the supplier's behavior in charge

## 7. Publish one resolved generation, then authorize its catalog

`ModelOfferingsManager._apply` compiles the resolved definitions through [Router.snapshot_with_model_list](../../litellm/router.py), setting `model_metadata_authoritative=True`. Native deployment records carry private typed provider policy, not an independent public policy registry

The [serving view and ASGI middleware](../../litellm/proxy/offering_router.py) publish one generation atomically and pin that generation for each HTTP request, including the streaming response lifetime. Subsequent requests can see a newer generation while the earlier request finishes on its original view. This is a generation-pinning contract, not a claim that the whole native Router and all shared clients are deeply immutable. The facade blocks external model-definition mutation while reusing native resources

The [shared model checks](../../litellm/proxy/auth/model_checks.py), native key/team/alias authorization and offering listing guard determine which selected public names the caller can see. Wildcard grants match live selected public Router names, not only old supplier registry entries. This does not broaden an exact key into access to other offerings. Alias lookup reuses the native listing/deployment context

[create_model_info_response](../../litellm/proxy/utils.py) constructs the public object through the [discovery projection](../../litellm/proxy/model_discovery.py). In authoritative external mode it does not perform a second unrelated catalog fallback after offering resolution. The intended unpinned bundled fallback has already happened in `_deployment`. Private credentials, deployment identity and routing policy are not public model metadata

## 8. Client options and the inference path

A client reads its authorized gateway `/v1/models` result. A generic client can present the exact effort list as choices or variants, keep context/input/output distinct, show an advertised default and use a request-default policy when constructing a request. It must keep unknown fields unknown instead of manufacturing a supplier-specific limit or effort ladder

Variant syntax and UI conversion belong to the client, not this LiteLLM PR. A client that rejects a requested effort variant when its catalog has no variants can do so before any inference request. This document does not claim to implement or test OpenCode's local validation or interactive UI, and no OpenCode source change is needed to define the gateway catalog contract

For inference, the public alias resolves to a selected native deployment. The Router applies request-default budget policy and dispatches through the selected provider adapter. The [OpenRouter chat adapter](../../litellm/llms/openrouter/chat/transformation.py) preserves explicit gateway `max`/`xhigh`, and the [Messages bridge normalization](../../litellm/llms/anthropic/pass_through/utils.py) does not reduce OpenRouter effort using unrelated backend flags. Native Responses and Responses-to-chat paths retain their API-specific request shapes

The offering guard merges the exact configured `only`/`allow_fallbacks: false` body and checks it again at deployment dispatch. Caller connection, deployment, provider, fallback, tier and parameter-dropping overrides cannot replace it. Pinned response-cache reads/writes are disabled to prevent output from another endpoint satisfying the request. No `require_parameters` setting or fallback supplier is silently added. Unsupported pinned API surfaces and native batches are rejected rather than bypassing selection

## 9. Refresh, failure and present limitations

Configuration polling defaults to five seconds, configurable from one to 300. Inventory polling defaults to 300 seconds, configurable from one to 86400. Known connection/base/credential or native-account changes invalidate the relevant prior state; changed inventory intervals or selected pinned-model requirements trigger fresh discovery. This is periodic discovery and atomic publication, not an instantaneous supplier query for every client request

Inventory HTTP requests have a five-second timeout, a two-megabyte response bound and redirects disabled. The manager bounds concurrent provider refreshes to eight; endpoint requests within a provider are sequential. Cache identity includes supplier/base/authentication, and endpoint retention additionally checks credential scope. Positive compatible/endpoint inventories use a 300-second cache TTL and compatible/endpoint failures a 60-second negative TTL; the manager's forced scheduled refresh bypasses cached results

A successful missing-model inventory removes an AUTO model. A failed fetch is not an empty inventory and retains last-known data only where identity permits. A successful empty endpoint inventory clears derived endpoint metadata without itself disabling an otherwise selected model. An initial endpoint failure leaves fields unknown rather than using unrestricted capacity. An invalid file edit or rejected candidate Router leaves the previous complete serving configuration in place

Important boundaries of the current implementation remain

- Automatic metadata is limited to fields the adapters recognize and the applicable bundled fallback can justify. It is not independent benchmarking, and unknown values can legitimately remain absent
- Raw Codex UI windows are retained internally without a generic physical-window guarantee. The Ultra default fallback is a normalization rule, not evidence of backend equivalence
- The [native ChatGPT Responses adapter](../../litellm/llms/chatgpt/responses/transformation.py) filters outgoing keys and currently excludes output-token budget fields. Do not promise that a configured budget is enforced on that backend merely because the shared helper can select it
- Pinning supports the current strict OpenRouter chat policy schema. Unknown routing controls, endpoint health/account-specific eligibility not represented by that schema, or an undocumented supplier-native effort mapping are not established by this metadata calculation
- An explicit override must be internally consistent and is not a deletion mechanism when set to null. A copied workaround can freeze AUTO fields until removed; it is not a required permanent policy
- Other clients may interpret catalog extensions differently. The regression evidence is in-process inventory, ASGI, Router and captured HTTP requests, not interactive client or paid supplier proof

The [endpoint inventory tests](../../tests/unit/llms/openrouter/test_model_info.py), [offering catalog/request tests](../../tests/unit/proxy/test_model_offerings.py), [budget tests](../../tests/unit/router_utils/test_model_request_defaults.py) and [native account tests](../../tests/unit/llms/chatgpt/test_model_info.py) provide concrete examples of these contracts without deployment-local configuration or credentials
