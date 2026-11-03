# Recipes

## A new failure in litellm_http

Say media downloads must reject a `Content-Type` outside an allow-list

1. Add a `Repr` variant with typed facts and the message, `#[error("media type {0:?} is not allowed")] DisallowedMediaType(String)`
2. Return it where the check fails, `return Err(Repr::DisallowedMediaType(content_type.to_owned()).into())`
3. Add an arm to every predicate's exhaustive `match`, the compiler lists them, the way its neighbor `DownloadTooLarge` answers. A media download happens before the provider call, so it is not in `dispatched()`. The caller chose the URL, so it is in `is_request()`
4. Leave both boundaries alone. With those answers the gateway returns 400 and Python raises `ValueError`
5. Test it with a mock server answering `Content-Type: application/x-msdownload` and assert `error.is_request()`

## A provider-specific failure

A second OCR provider needs a missing-key error whose text names its environment variable and how to set it

1. Look for an existing provider-neutral variant whose template fits. `litellm_auth::Error::MissingApiKey { provider, environment_variable }` renders `Missing {provider} API Key - Set ...`
2. If none fits, find what varies between providers with the same failure, here the environment variable and the guidance after the dash, and add one neutral variant with those as fields, `#[error("Missing {environment_variable} - {guidance}")] MissingCredential { environment_variable: &'static str, guidance: &'static str }`
3. Pass the provider's values from its transformation. The next provider passes its own and adds no variant, and `is_missing_credential()` lists `MissingCredential { .. }` once

## Wrapping a new lower crate in litellm_inference

Say routes start calling `litellm_cache` and its errors must surface

1. Add one variant, `#[error(transparent)] Cache(#[from] litellm_cache::Error)`
2. Extend every predicate's `match` with an arm that calls through to the cache error's own predicates, adding a predicate there if it has none
3. Do not copy the cache error's variants into `litellm_inference::Error`

## A foreign error that needs context

Reading a file fails with `io::Error`, and the message must name the file

1. A variant with the context and the cause, `Read { path: PathBuf, source: io::Error }` with `#[error("could not read {}", path.display())]`
2. Build it with `map_err(|source| Repr::Read { path: path.clone(), source })`, not a `From<io::Error>`
3. If the error must be `Clone`, the crate's error is opaque or the field is `litellm_error::Source`

## Migrating a stringly error

Call sites look like `Error::Network(error.to_string())`, `Error::Request(format!(...))` or `.map_err(|_| Error::Unavailable)`

1. List every call site of the string variant and group them by what actually failed
2. Each group becomes a variant holding the typed cause or facts. Groups no caller tells apart share one
3. Where a foreign error was stringified with classification logic around it, move that logic into the `From` and turn the call sites into `?`
4. Delete the string variant. If it cannot be deleted yet, nothing new may construct it
5. Re-run the boundary tests, since messages a client sees may change

## Moving a crate to source chains

A variant reads `#[error("invalid JSON: {0}")] Json(#[from] serde_json::Error)` or stores `cause` and renders `{cause}`

1. Drop the cause from the template, `#[error("invalid JSON")]`
2. Make the cause the source, `#[from]`, `#[source]` or a field named `source`
3. Find what renders this error for someone else and make it use `litellm_error::Report`
4. Update tests that asserted the cause in `to_string()` to assert it in the report

## Making a public enum opaque

Downstream crates match nested variants, or a variant needs a non-`Clone` cause

1. Rename the enum to `pub(crate) enum Repr` and add `pub struct Error(Arc<Repr>)` with `#[error(transparent)]` and `From<Repr>`
2. For each downstream `match`, find the question it asks and add a predicate. Replace the `match` with the predicate
3. For each downstream construction, add a constructor taking typed data, or move the failure into this crate
4. Swap test assertions to predicates and add the `#[cfg(test)] repr()` accessor for in-crate tests
5. Implement `PartialEq` with `Arc::ptr_eq` only if a wrapping enum's derive needs it
