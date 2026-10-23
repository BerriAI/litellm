# Credential and resource provenance

- `Sourced<T>` carries the host-assigned `InputSource` alongside a value until validation
- `CredentialBoundary::new(&sources)` accepts a complete request-owned credential bundle or a host-owned bundle, never a mixture or an empty bundle
- `CredentialBoundary::host_only(&sources)` additionally rejects request-owned credentials, for ambient identities, assertion providers and host file references
- `boundary.validate_resources(sources)` rejects every request-controlled resource when credentials come from deployment or environment configuration
- `boundary.source()` preserves request ownership, otherwise preferring deployment over environment provenance
- Failures return `UntrustedCredentialContext` without values or secrets. Providers map it into their existing public error contract

## Provider contract

- Provider-wide code selects the required credential components and protected resource inputs. Infrastructure does not infer them from parameter names
- Include endpoint, credential issuer, audience and resource project or space when the provider supports them, including every accepted alias and nested spelling
- Check all supplied protected inputs before precedence or alias resolution can discard one. Then validate the effective credential bundle and resource values before credential acquisition or provider IO
- Missing deployment values do not authorize request overrides. Even equal request and deployment values remain request-controlled until explicitly authorized by the host
- A fully request-owned credential bundle can select request resources only when that provider permits caller credentials. Host-only identities always use `host_only`
- This check validates provenance, not URL safety, credential completeness or permissions at the upstream service. Providers still validate those separately

## Host contract

- A trusted in-process SDK caller constructs deployment-sourced values, and environment lookup constructs environment-sourced values
- A Router keeps per-call overrides separate from deployment credentials until validation. It must not trust a caller-supplied internal dictionary as deployment configuration
- Python proxy and Rust HTTP hosts label all HTTP-supplied fields as request inputs, including aliases, nested data and connection headers. Never deserialize trusted source labels or an authorization bool from HTTP JSON
- Hosts retain provenance when projecting into native `Sourced<T>` inputs. Source maps are an internal host interface, not a public request field
- An existing explicit admin opt-in may authorize the specific configured field. The host makes that decision from server-owned configuration before constructing the native inputs, never from the request body
- Python proxy opt-ins remain `general_settings.allow_client_side_credentials` and deployment `configurable_clientside_auth_params`. This primitive does not change those checks or grant either permission
- New tokenization hosts must supply this contract themselves. Existing OCR wire inputs and Python projection remain unchanged in this extraction

## Existing consumers

- Azure native auth validates client-secret identities against scope and authority before constructing its token acquirer, and keeps ambient identities host-only
- Azure AI provider-wide destination policy validates endpoint provenance for OCR credentials, preserving existing errors and accepted source combinations
