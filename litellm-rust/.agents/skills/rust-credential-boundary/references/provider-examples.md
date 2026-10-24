# Provider examples

Each example shows which sources the provider feeds into `CredentialBoundary::new` and `validate_resources`, and how it maps the failure onto its existing error

## Accepted combinations

| credentials | resource | result |
| --- | --- | --- |
| all request | request | ok, the caller spends its own credential |
| all request | deployment or environment | ok |
| deployment, environment or both | deployment or environment | ok, `source()` prefers `Deployment` |
| deployment or environment | request | rejected |
| request mixed with deployment or environment | any | rejected |
| empty | any | rejected, an empty bundle authorizes nothing |

## Azure native client-secret auth

`crates/auth-azure/src/native.rs`. Three credential components, two protected resources, and the message Python parity pins

```rust
let identity_sources = [tenant_id.source(), client_id.source(), client_secret.source()];
let boundary = CredentialBoundary::new(&identity_sources).map_err(|_| mixed_sources())?;
boundary
    .validate_resources(std::iter::once(scope.source()).chain(authority.as_ref().map(Sourced::source)))
    .map_err(|_| mixed_sources())?;
Ok(boundary.source())
```

Ambient identities and client assertions in the same file go through `host_only`, so a request can never pick a managed identity or a federated token file

```rust
fn trusted_only(sources: &[InputSource]) -> Result<InputSource, Error> {
    CredentialBoundary::host_only(sources)
        .map(|boundary| boundary.source())
        .map_err(|_| mixed_sources())
}
```

## Azure AI OCR

`crates/llms/src/azure_ai/common_utils.rs`. One api-key credential and one protected endpoint. `None` for the endpoint means the request did not supply one, and the boundary accepts an empty resource list

```rust
pub(crate) fn validate_destination(
    endpoint_source: Option<InputSource>,
    credential_source: InputSource,
) -> Result<(), litellm_auth::Error> {
    CredentialBoundary::new(&[credential_source])
        .and_then(|boundary| boundary.validate_resources(endpoint_source))
        .map_err(|_| litellm_auth::Error::InvalidConfiguration(
            "host credentials cannot be sent to a request-controlled Azure endpoint".into(),
        ))
}
```

## Vertex AI OCR

`crates/llms/src/vertex_ai/ocr/common_utils.rs` still hand-rolls its check and rejects a request-controlled `api_base` regardless of where the credential came from. That is stricter than the boundary because Vertex credentials are a service account the host holds, so it is a `host_only` consumer waiting to be migrated. When touching it, pass `vertex_project` and `vertex_location` as protected resources too

## Watsonx

The Watsonx layer of the tokenization stack, in `crates/llms/src/watsonx/common_utils.rs`. Four protected resources, and the sources are read from both the deployment params and the per-call optional params before precedence resolution can drop a duplicate

```rust
let boundary = CredentialBoundary::new(&sources)
    .map_err(|error| Error::InvalidRequest(error.to_string().into()))?;
boundary
    .validate_resources(
        input.api_base.into_iter().map(Sourced::source).chain(
            input.deployment_params.iter().chain(input.optional_params)
                .filter(|(name, _)| matches!(name.as_str(), "project_id" | "watsonx_project" | "space_id" | "api_version"))
                .map(|(_, value)| value.source()),
        ),
    )
    .map_err(|error| Error::InvalidRequest(error.to_string().into()))?;
```

Note the alias pair `project_id` and `watsonx_project`. Both are checked even though only one wins when the request is built
