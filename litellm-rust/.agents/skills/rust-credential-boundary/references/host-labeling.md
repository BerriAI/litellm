# Host labeling

The auth crates never learn who called them. Every host hands them values plus a map from field name to `InputSource`, and the boundary reads only that map. OCR is the one route all three hosts run, so the examples use it

## The shared shape

`crates/inference-ocr/src/wire.rs`

```rust
pub struct OcrWireRequest<D = Value> {
    pub model: String,
    pub api_key: Option<SecretValue>,
    pub api_base: Option<String>,
    pub optional_params: Map<String, Value>,
    #[serde(default)]
    pub input_sources: BTreeMap<String, InputSource>,
    ...
}
```

A missing key in `input_sources` means `InputSource::default()`, which is `Deployment`. The whole property reduces to the host putting `Request` into that map for everything the caller supplied

## Python proxy

`litellm/proxy/litellm_pre_call_utils.py` attaches a record of what came over HTTP to every call

```python
data["proxy_server_request"] = {
    "url": ..., "method": ..., "headers": ...,
    "body": <parsed JSON body, filled after the admin-field strip>,
    "credential_fields": ("api_key",),
}
```

`request_input_sources` in `crates/python-bridge/src/marshal.rs` intersects the provider's consumed param names with the keys present in `body` or `credential_fields` and emits `Request` for each hit. Given a deployment with `AZURE_API_KEY` in the environment and this request

```json
POST /v1/ocr
{"model": "azure_ai/doc-intel", "document": {...}, "api_base": "https://attacker.example"}
```

the bridge produces

```
api_key        = None   (resolved later from env as Sourced(key, Environment))
api_base       = "https://attacker.example"
input_sources  = {"api_base": Request}
```

and Azure AI's `validate_destination(Some(Request), Environment)` fails. The Python blocklist in `is_request_body_safe` already rejects `api_base` at the edge, so the Rust check is the second fence, and the only fence for fields the blocklist does not list such as `scope`, `authority`, `project_id` and `space_id`

## Python SDK and Router

A direct `litellm.ocr(model=..., api_base=..., api_key=...)` has no `proxy_server_request` in kwargs, so the map is empty and everything is `Deployment`. In-process SDK code is the trusted operator, so that is correct. The Router merges deployment params into kwargs before the bridge sees them, which means it does not yet keep per-call overrides apart from deployment credentials. Until it does, the Router path relies on the proxy blocklist alone

## Native gateway

`crates/gateway-inference/src/ocr.rs`. Axum parses the body, `resolve_deployment` picks a configured `Deployment`, and the handler builds the connection from the deployment only

```rust
OcrConnectionInputs {
    api_key: deployment.api_key.clone().map(SecretValue::new),
    api_base: deployment.api_base.clone(),
    timeout: deployment.timeout,
    ..Default::default()
}
```

A body `api_base` cannot reach credentials because the handler never copies it out of the body. The same request as above yields

```
api_key        = Sourced(deployment key, Deployment)
api_base       = Sourced(deployment base, Deployment)
input_sources  = {}
```

The leftover body fields do flow into `optional_params` with no entry in `input_sources`, so for Vertex OCR a body-supplied `vertex_project` is labeled `Deployment` by default. Nothing exploits that today because those fields do not retarget credentials, but it is the unlabeled-request-field bug described in the skill, and new gateway routes must not copy it

A gateway route whose provider accepts body resources, such as Watsonx with `project_id`, `space_id` and `api_version`, has to label them itself before calling the provider

```rust
input_sources: body.keys().map(|name| (name.clone(), InputSource::Request)).collect()
```
