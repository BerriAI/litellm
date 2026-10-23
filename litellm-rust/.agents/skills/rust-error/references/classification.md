# Classification

## Ask, don't match

Downstream crates ask a lower error questions instead of matching its nested variants. `litellm_http::Error` answers these

| Predicate | Caller decides |
|---|---|
| `dispatched()` | whether retrying elsewhere could repeat a billed call |
| `is_request()` | the caller's input is at fault, so 400 and no reroute |
| `is_response()` | the provider answered with something we cannot use |
| `upstream_response()` | the provider's own status and body to pass through |
| `status()` | an HTTP status to report |
| `is_connect()` | the connection was never made, for a host's own connect-error mapping |
| `tls_source()` | which TLS setting to blame in a config error |
| `is_computed_header()` | the caller forwarded a header the signer owns, which `litellm_inference::Error` turns into `Unsupported` |

Upper types add the questions their callers ask. `litellm_inference::Error` and the OCR error answer `is_missing_credential()` and `is_unsupported()`, which the gateway asks instead of reaching into `Auth` or `Unsupported`

```rust
// reaches through two crates' variants and breaks when either changes
Self::Route(InferenceError::Auth(litellm_auth::Error::MissingApiKey { .. })) => StatusCode::UNAUTHORIZED,

// asks the question it cares about
Self::Route(error) if error.is_missing_credential() => StatusCode::UNAUTHORIZED,
```

A predicate is an exhaustive `match` with no `_` arm, never `matches!`, so a new variant does not compile until it has answered every question on purpose

```rust
pub fn is_request(&self) -> bool {
    match &*self.0 {
        Repr::Request(_) | Repr::BlockedUrl | Repr::DownloadTooLarge => true,
        Repr::Connect(_) | Repr::Network(_) | Repr::Timeout(_) | Repr::Status { .. } => false,
        // ... every other variant, listed by name
    }
}
```

Add a predicate when a real caller needs a new distinction, and name it after the fact the caller needs, in the caller's words, not after the variant's spelling. An upper type's predicates call through to the nested error, as in `Self::Http(error) => error.is_request()` and `Self::Auth(error) => error.is_missing_credential()`

A public enum may be matched one level down by the type that owns the variant, such as the gateway matching its own `Unsupported` or the OCR bridge matching `FileRead`. Going a second level down, into a variant of a variant, is what to avoid, the gateway used to do that for `MissingApiKey` and `Unsupported`

## Facts, not policy

A predicate reports what happened. The decision to retry, reroute or fall back depends on things the error cannot know, such as whether the request is idempotent, whether its body can be replayed, the deadline and the attempt budget. That decision lives with the caller. `dispatched()` exists for the router's reroute decision, so every transport failure answers it even before that caller is wired

A timeout is the main trap. A request that timed out may already have reached the provider and been billed, so `Timeout` answers `dispatched() == true` and is not safe to retry as if it were a connect failure. Only a failure that provably happened before any byte reached the provider answers `false`: connecting, building or signing the request, and media downloads, which fetch the caller's document before the provider call is made

```rust
pub fn dispatched(&self) -> bool {
    match &*self.0 {
        Repr::Network(_) | Repr::Timeout(_) | Repr::Status { .. } => true,
        Repr::Connect(_) | Repr::RequestBody(_) | Repr::Download(_) | Repr::BlockedUrl => false,
        // ... every other variant, listed by name
    }
}
```

When the transport cannot tell whether a request reached the provider, it must answer `true`. Wrongly answering `false` risks a duplicate billed call, wrongly answering `true` only costs a reroute

Keep one predicate per question. `status()` currently mixes the provider's status, the media download's status and a synthesized 408 for timeouts, so a caller that wants "did the provider answer" must use `upstream_response()` instead. Do not add more meanings to an existing predicate, add a new one

## Equality

`Arc<Repr>` errors compare with `Arc::ptr_eq`, so a clone equals its original and two separately built errors never compare equal. That exists only so a wrapping enum such as `litellm_inference::Error` can keep its `PartialEq` derive. Do not give a new type pointer equality for any other reason, and do not rely on it in tests. Assert predicates instead, see [testing](testing.md)
