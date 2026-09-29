# Typed provider endpoints

Status: Draft for implementation planning

Scope: Rust provider endpoint definitions and outbound URL resolution, preserving successful Python SDK override behavior

Source revision inspected: `47164759e8a`, on 2026-09-28

This handoff proposes APIs and migration steps. The Rust types and macros shown below do not exist yet, except for the existing `ApiUrl` and `MistralBatchEndpoint` names

## Problem and evidence

Provider operations should select typed endpoints. Shared URL mechanics should produce a complete, validated destination once, and that distinction should survive through transport

| Current code | Issue or existing behavior |
| --- | --- |
| [`MistralBatchEndpoint`](../../crates/llms-types/src/providers/mistral/batches.rs) | Serde represents batch targets as paths such as `/v1/ocr`. Those are relative to an origin, not complete URLs |
| [`ApiUrl`](../../crates/core-utils/src/url_utils.rs) | Already distinguishes `Base` and `Complete`, but infers meaning from arbitrary suffix overlap and exposes the completed result as `String` |
| [Cohere OCR](../../crates/llms/src/cohere/ocr/transformation.rs) | Already honors `connection.api_base`, but repeats fallback selection, parsing, scheme checks, and route completion |
| [OpenAI Responses](../../crates/llms/src/openai/responses/transformation.rs) | HTTP appends `/responses` as text. WebSocket construction separately splits queries, changes schemes, and encodes values |
| [Core OCR dispatch](../../crates/core/src/ocr/provider_config.rs) | `passthrough_response` compares provider-specific endpoint paths inside core |
| [Core OCR preparation](../../crates/core/src/ocr/prepare.rs) | Selects provider-specific base-URL environment variables inside orchestration |
| [Outbound requests](../../crates/http/src/outbound.rs) | Carry a string URL, so downstream code cannot distinguish a completed destination from a base |

An endpoint enum alone does not settle what `api_base` means. A root, gateway prefix, API-version prefix, and full endpoint are different inputs even when each is a valid `url::Url`

## Decision and invariants

Represent a wire path, an explicitly chosen mount, and a complete destination as different types. Keep `url::Url` as the underlying URL representation. Endpoint resolution returns a completed wrapper, with access to `Url` when transport needs it

Static provider paths have one authoritative declaration. Consumers use a typed path or endpoint operation, without recovering the path by serializing the enum to JSON

A mount is an origin plus an optional gateway prefix. Resolution appends the entire provider route once. An exact destination already includes its path and query. It bypasses route and query construction

Only an unfinished URL can receive a route or generated query parameters. Completed URLs expose no mutation or completion methods. Invalid input returns an error through the existing public error mapping

Typestate prevents accidental misuse through this API. It cannot discover whether an arbitrary caller-supplied string was intended as a mount. Explicit constructors and provider-owned compatibility normalization supply that meaning

An endpoint declaration describes an upstream operation. It does not establish model capability, batch eligibility, or the existence of a LiteLLM adapter for that operation

## Ownership and file layout

Use the existing crates and one endpoint module per provider to begin with. The following paths are relative to `litellm-rust/` and include proposed files

```text
crates/llms-types/src/endpoint.rs
crates/llms-types/src/providers/mistral/batches.rs
crates/core-utils/src/url_utils.rs
crates/llms/src/base_llm/endpoint.rs
crates/llms/src/cohere/endpoints.rs
crates/llms/src/mistral/endpoints.rs
crates/llms/src/azure_ai/endpoints.rs
crates/http/src/outbound.rs
```

`llms-types::endpoint` owns `EndpointPath` and the macro for path-valued wire enums. This is a data contract shared by serialization and request construction. It has no defaults, environment lookup, HTTP client, or dependency on execution crates

`core-utils::url_utils` owns URL states, validation, path-segment encoding, and query mechanics. Extend the existing module rather than adding a second URL builder. Move its URL error definition into the crate's `error.rs` when changing it, following the workspace error rules

`llms::base_llm::endpoint` owns the common endpoint contract, `ResolvedEndpoint`, and the runtime declaration macro. Provider `endpoints.rs` modules own operations, default origins, legacy base interpretation, and provider-specific query policy. Keep existing credential-source precedence and provenance intact

Keep provider endpoint declarations together across OCR, Messages, Responses, and other transformations. If a provider has several distinct services, split into `endpoints/<service>.rs` when needed. Avoid a workspace-wide enum containing every provider operation

Core selects an adapter and consumes its result. It does not select provider URL suffixes or parse provider paths. HTTP consumes a method and completed URL without importing provider enums or `llms`

## Proposed API

### URL states and transport access

Evolve `ApiUrl` to name the actual input contract. Signatures below describe the intended interface and omit implementations

```rust
pub struct Mount;
pub struct Complete;

pub struct ApiUrl<State> {
    url: url::Url,
    state: std::marker::PhantomData<State>,
}

pub enum EndpointTarget {
    Mount(ApiUrl<Mount>),
    Exact(ApiUrl<Complete>),
}

impl ApiUrl<Mount> {
    pub fn parse_mount(value: &str) -> Result<Self, ApiUrlError>;
    pub fn resolve(&self, path: EndpointPath) -> Result<ApiUrl<Complete>, ApiUrlError>;
}

impl ApiUrl<Complete> {
    pub fn parse_exact(value: &str) -> Result<Self, ApiUrlError>;
    pub fn as_url(&self) -> &url::Url;
    pub fn into_url(self) -> url::Url;
}
```

The query-bearing resolution operation accepts typed query decisions and finishes path and query together. The simple `resolve` above covers routes without generated query parameters. There is no `append_query_pairs` on `Complete`

Keep URL fields private. Do not implement `DerefMut`, `AsMut<Url>`, or a conversion from `Complete` to `Mount`. Native prepared requests and `OutboundRequest` retain `ApiUrl<Complete>` until the HTTP client needs `into_url()`. Foreign interfaces may serialize a destination as a string, but must recover `Complete` on return rather than rerunning provider completion

HTTP validation requires a hierarchical URL with a host and an HTTP(S) scheme. WebSocket support gets its own validated transport wrapper when migrated. Its provider resolver selects WS(S) before completion, with no scheme-flipping method on a completed HTTP URL. Decisions about fragments and userinfo are listed under open decisions because stricter rejection could change legacy acceptance

### Wire paths and the existing Mistral enum

`EndpointPath` is a small immutable value with private storage, validated static construction, and a segment iterator. Its initial contract is an absolute path reference containing static route segments, with no scheme, authority, query, fragment, percent escapes, empty segments, or dot segments. Dynamic identifiers use a separate checked segment input in the runtime resolver

Use a small `macro_rules!` declaration to generate both Serde names and typed accessors from the same literals. Proposed syntax, showing a subset of the existing variants:

```rust
wire_endpoint_enum! {
    pub enum MistralBatchEndpoint {
        ChatCompletions => "/v1/chat/completions",
        Embeddings => "/v1/embeddings",
        Ocr => "/v1/ocr",
    }
}
```

Expansion preserves the current derives, optional schema feature, and exact serialized values. It adds `const fn path(self) -> EndpointPath`. Validate literals during constant evaluation, with diagnostics identifying the invalid declaration. The implementation keeps every existing enum variant

The normal API has no `as_str`, `Display`, or `Into<String>` for this path. Serialization still emits the provider-required string. `Recognized<MistralBatchEndpoint>` continues preserving unknown wire values, but unknown strings never become executable endpoints automatically

Do not add `url()` with a built-in host to the batch enum. A full URL depends on deployment settings, and `llms-types` must remain independent of those settings

### Runtime declarations and generated resolution

Use a second narrow macro in `llms` for executable endpoint metadata and the shared resolver call. It accepts either a checked static literal or an existing typed path expression

```rust
provider_endpoints! {
    pub enum CohereEndpoint {
        Parse => POST "/v2/parse",
    }
}

provider_endpoints! {
    pub enum MistralEndpoint {
        Ocr => POST path(MistralBatchEndpoint::Ocr.path()),
    }
}
```

Generate an exhaustive operation-to-method/path mapping and implement a shared `ProviderEndpoint` trait whose entry point is `resolve(&self, target: &EndpointTarget) -> Result<ResolvedEndpoint, Error>`. `ResolvedEndpoint` holds the HTTP method and `ApiUrl<Complete>`, with read-only access and an ownership-taking decomposition for core. Use the existing HTTP method type rather than inventing another method enum

The intended caller is:

```rust
let endpoint = CohereEndpoint::Parse.resolve(&target)?;
let url: &url::Url = endpoint.url().as_url();
```

This provides the requested method returning `url::Url` through the completed wrapper. Returning a bare URL from every transformation would discard the base-versus-complete guarantee too early

Keep macro expansion small and delegate mechanics to ordinary functions. The wire macro cannot depend on the runtime macro because the crate dependencies run in the opposite direction. Neither macro handles environment lookup, credentials, defaults, polling, or arbitrary path-template interpolation

Dynamic operations initially use ordinary enums and handwritten implementations of the same endpoint contract. For example, `RetrieveBatch { id: BatchId }` requires an ID at construction. Azure and Vertex operations receive typed deployment, project, location, and model inputs. Do not make the macro a template language just to cover these cases

### URL construction rules

Preserve the mount's existing encoded prefix and append checked segments. Do not rebuild the prefix from `path_segments()` as if it were decoded input. Dynamic values are unencoded identifiers and are encoded once as individual segments. A slash inside an ID stays inside that segment. Reject dot-only dynamic identifiers before composition, and test percent signs, spaces, Unicode, slashes, and backslashes

Use the existing `url` library for URL parsing and segment mutation. `Url::join` implements reference resolution, so a leading slash replaces the base path and a missing trailing slash changes how the last base segment is treated. It is not the mount-append operation required here. See the versioned [`Url::join` documentation](https://docs.rs/url/2.5.8/url/struct.Url.html#method.join)

Treat provider action suffixes such as `:analyze` and `:rawPredict` as explicit route construction using checked identifiers. Do not allow an arbitrary formatted URL template to bypass segment validation

Provider policy must choose whether a generated query key supplies a default, replaces existing values, or rejects a conflict. Shared mechanics apply that decision once and preserve unrelated query pairs and repeated keys. Preserve raw query encoding when no query change is required. Exact targets bypass generated parameters entirely, so any mandatory query validation must fail clearly rather than silently changing the URL

Resolve regions and other hostname inputs through provider validation before forming a default origin. Explicit base overrides retain their existing precedence. A server-returned polling URL is already complete and must not receive another operation path. Preserve the current credential and polling-origin rules

## Compatibility

Keep Python's public `api_base`, `base_url` aliases where supported, and `custom_endpoint` behavior. Introducing typed internal targets does not require adding a public setting

The compatibility boundary is the provider adapter entry point, after effective configuration and its source have been selected. Core invokes it, while provider code interprets the string. Do not put provider suffix recognition into a generic Python decoder or `llms-types`

| Input and operation | Required interpretation |
| --- | --- |
| Explicit Rust mount `https://gateway.test/team` plus Cohere Parse | Append the complete route, yielding `/team/v2/parse` |
| Explicit Rust mount ending in `/v2` | Treat `/v2` as part of the declared mount, without guessing that it is an API version |
| Legacy Cohere OCR base ending in `/v2` | Preserve current version-prefix completion, yielding `/v2/parse` |
| Legacy Cohere OCR base ending in `/v2/parse` | Preserve the complete destination |
| Exact destination `https://gateway.test/custom?tenant=a` | Preserve path and query, without adding `/v2/parse` |
| OpenAI-compatible `custom_endpoint=True` | Preserve the complete destination |
| Legacy OpenAI Responses base ending in `/v1` | Preserve `/v1/responses` |
| Legacy OpenAI Responses base ending in `/v2` | Preserve `/v2/responses`, without rewriting the version to the provider default |

Current Python behavior is visible in [OpenAI-compatible URL handling](../../../litellm/llms/openai_like/common_utils.py), [Responses URL handling](../../../litellm/llms/openai/responses/transformation.py), and [Anthropic Messages URL handling](../../../litellm/llms/anthropic/pass_through/messages/transformation.py). [Python OCR](../../../litellm/ocr/dispatch.py) passes `api_base` into Rust in the inspected checkout

Normalize each legacy input once for the selected operation. Known complete URLs become `Exact`. Known prefixes use provider-specific rules. When legacy behavior cannot be expressed as a canonical mount, the compatibility adapter constructs the legacy destination and validates it as `Complete`. The normal resolver does not perform maximum-suffix-overlap inference

An exact target is scoped to one request operation. Do not cache it as a provider-wide base and reuse it for a different operation. Store the original deployment input and independently normalize it when another operation is selected

Preserving successful overrides is the baseline. Fixing existing malformed construction, such as a duplicated `/responses`, needs an explicit regression case and a documented behavior change. An unfamiliar legacy path might be a valid gateway prefix, so recognition must not silently strip arbitrary suffixes

Retain `Sourced` provenance through normalization. Destination typing must not make request-supplied URLs eligible for deployment-owned credentials. Hooks that change a URL invalidate its previous validation. Validate the final URL before cache identity, signing, and sending, using the same URL value for all three. Do not rewrite signed query strings after signing

## Implementation and verification

### First slice

Implement `EndpointPath`, the URL state operations, and a handwritten Cohere endpoint first. Migrate Cohere OCR preparation through `OutboundRequest` so the completed type survives to transport. Keep other adapters working through explicit complete-URL conversion at their existing finalization boundary. Do not convert the new typed result immediately back to a string just to fit the old shared API

Add the wire macro, preserving Mistral serialization. Migrate Mistral OCR to reuse `MistralBatchEndpoint::Ocr.path()`. Extract the runtime macro only after both static providers have exercised the same ordinary implementation

### Dynamic and shared behavior

Migrate one Azure or Vertex operation with dynamic identifiers and query parameters. This checks that the abstraction handles actual provider variation before spreading it across the workspace. Move core's OCR endpoint recognition behind a provider-owned operation matcher using the same route definitions

When migrating operations with methods other than POST, propagate the declared method through transport, signing, and cache identity together. The current `OutboundRequest` always posts and its signer input omits the method, so adding method metadata alone is insufficient

Migrate Messages, Responses, WebSockets, and batch polling incrementally. Remove old provider URL builders and URL constants as their callers move. Retire `Base::complete_path` and `into_string` after no native preparation path needs them. Keep runtime defaults inside the provider's endpoint module

### Acceptance criteria

| Area | Observable proof |
| --- | --- |
| Typestate | Compile-fail examples cannot resolve from `Complete` or mutate its URL through the wrapper. Include a passing companion so an unrelated import failure cannot satisfy the check |
| Wire contracts | Existing known variants and unknown `Recognized` values round-trip unchanged, with and without the schema feature. A generated typed path agrees with the enum's serialized value |
| Resolution | Injected mounts preserve gateway prefixes. Exact destinations receive no suffix or generated query. Resolving several operations from one mount does not accumulate paths |
| Compatibility | Existing root, version-prefix, complete-endpoint, blank-value, and environment precedence cases retain their successful behavior through the public preparation path |
| Encoding | Dynamic IDs cannot introduce path/query structure, `%` is encoded once as data, and existing encoded prefixes are not double-encoded |
| Queries | Provider-selected duplicate policy works, unrelated pairs survive, and unchanged exact queries retain their encoded form |
| Transport | An injected server or client observes the declared method and final path/query. Signing and cache identity consume the same final destination after hooks |
| Ownership | Core delegates endpoint recognition to the adapter, verified by behavior rather than source-text assertions |

Use synthetic hosts and injected defaults for resolver tests. Assert invariants owned by LiteLLM, including agreement between wire values and endpoint paths. If an external provider fact must be pinned, cite its source and inspection date beside the assertion. Tests of new URL behavior belong in `core-utils`, serialization tests in `llms-types`, provider policy tests in `llms`, and orchestration/transport tests in their respective crates

Run the affected crate tests and schema-feature tests for each slice, then the repository's required checks. Capture a real-provider request through the local proxy for an eventual implementation PR's proof of fix. This documentation change makes no claim of a runtime fix or passing implementation tests

## Open decisions

The recommended compatibility policy preserves successful legacy overrides while fixing demonstrated malformed URLs in explicit regression changes. The exact list of legacy suffix rules still needs to be recorded per provider before migrating it. Avoid a global rule that guesses all existing `api_base` meanings

Decide how legacy URLs with fragments, userinfo, or unusual query encoding should map into strict native types before changing their acceptance. `Url` parsing normalizes some spelling, so exact means no route/query composition, not byte-for-byte preservation of the original input string. Signed URL consumers need verification against the URL representation actually sent by transport

Keep the first runtime macro limited to static routes. Add dynamic macro syntax only if the first dynamic migration shows repeated code that ordinary typed constructors do not handle clearly. Public Python settings and a global provider endpoint registry are outside this proposal
