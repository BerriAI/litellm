# Recipes

## A new failure in litellm_http

Say media downloads must reject a `Content-Type` outside an allow-list

1. Add a `Repr` variant with typed facts and the message, `#[error("media type {0:?} is not allowed")] DisallowedMediaType(String)`
2. Return it where the check fails, `return Err(Repr::DisallowedMediaType(content_type.to_owned()).into())`
3. Answer every predicate on purpose, the way its neighbor `DownloadTooLarge` does. The download request went out, so it is in `dispatched()`. The caller chose the URL, so it is in `is_request()`
4. Leave both boundaries alone. With those answers the gateway returns 400 and Python raises `ValueError`
5. Test it with a mock server answering `Content-Type: application/x-msdownload` and assert `error.is_request()`

## A provider-specific failure with a parity message

Python raises `Missing REDUCTO_API_KEY - set it in the environment or pass api_key to litellm.ocr()/litellm.aocr()` and a test pins the text. The OCR error carries this today as `MissingReductoApiKey`, which is the shape to avoid

1. Look for an existing provider-neutral variant whose template produces the exact text. `litellm_auth::Error::MissingApiKey { provider, environment_variable }` renders `Missing {provider} API Key - Set ...`, which does not match
2. Find what varies between providers with the same failure. Here it is the environment variable and the guidance after the dash
3. Add one neutral variant with those as fields, `#[error("Missing {environment_variable} - {guidance}")] MissingCredential { environment_variable: &'static str, guidance: &'static str }`, and pass Reducto's values from the Reducto transformation
4. The next provider with the same failure passes its own values and adds no variant. Boundaries that mapped `MissingReductoApiKey` to 401 now match `MissingCredential { .. }` once

## Wrapping a new lower crate in litellm_inference

Say routes start calling `litellm_cache` and its errors must surface

1. Add one variant, `#[error(transparent)] Cache(litellm_cache::Error)`, with `#[from]` if no context is needed
2. Extend `is_request()` and `upstream_response()` with an arm that calls through to the cache error's own predicates, adding a predicate there if it has none
3. Do not copy the cache error's variants into `litellm_inference::Error`

## A foreign error that needs context

Reading a file fails with `io::Error`, and the message must name the file

1. A variant with the context and the cause, `Read { path: PathBuf, cause: io::Error }` with `#[error("could not read {}: {cause}", path.display())]`
2. Build it with `map_err(|cause| Repr::Read { path: path.clone(), cause })`, not a `From<io::Error>`
3. If the error must be `Clone`, the crate's error is opaque or the field is `Arc<io::Error>`

## Migrating a stringly error

Call sites look like `Error::Network(error.to_string())` or `Error::Request(format!(...))`

1. List every call site of the string variant and group them by what actually failed
2. Each group becomes a variant holding the typed cause or facts. Groups no caller tells apart share one
3. Where a foreign error was stringified with classification logic around it, move that logic into the `From` and turn the call sites into `?`
4. Delete the string variant. If it cannot be deleted yet, nothing new may construct it
5. Re-run the boundary tests, since messages a client sees may change

## Making a public enum opaque

Downstream crates match nested variants, or a variant needs a non-`Clone` cause

1. Rename the enum to `pub(crate) enum Repr` and add `pub struct Error(Arc<Repr>)` with `#[error(transparent)]` and `From<Repr>`
2. For each downstream `match`, find the question it asks and add a predicate. Replace the `match` with the predicate
3. For each downstream construction, add a constructor taking typed data, or move the failure into this crate
4. Swap test assertions to predicates and add the `#[cfg(test)] repr()` accessor for in-crate tests
5. Implement `PartialEq` with `Arc::ptr_eq` only if a wrapping enum's derive needs it
