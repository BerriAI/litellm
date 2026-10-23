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
| `tls_source()` | which TLS setting to blame in a config error |

```rust
// reaches through two crates' variants and breaks when either changes
Self::Route(InferenceError::Transport(TransportError::Http { status, .. })) => status,

// asks the question it cares about
Self::Route(error) if let Some((status, _)) = error.upstream_response() => status,
```

Add a predicate when a real caller needs a new distinction, and name it after what the caller decides, not after the variant. An upper type's predicates call through to the nested error, as in `Self::Http(error) => error.is_request()`

A public enum may still be matched where matching is the contract, such as `litellm_auth::Error::MissingApiKey` mapping to 401. Matching is fine at the top of the variant tree, reaching through a nested error is what to avoid

## Facts, not policy

A predicate reports what happened. The decision to retry, reroute or fall back depends on things the error cannot know, such as whether the request is idempotent, whether its body can be replayed, the deadline and the attempt budget. That decision lives with the caller

A timeout is the main trap. A request that timed out may already have reached the provider and been billed, so `Timeout` answers `dispatched() == true` and is not safe to retry as if it were a connect failure. Only a failure that provably happened before any byte left, such as connect or request building, answers `false`

```rust
pub fn dispatched(&self) -> bool {
    matches!(*self.0, Repr::Network(_) | Repr::Timeout(_) | Repr::Status { .. } | ..)
}
```

When the transport cannot tell whether a request went out, it must answer `true`. Wrongly answering `false` risks a duplicate billed call, wrongly answering `true` only costs a reroute

Keep one predicate per question. `status()` currently mixes the provider's status, the media download's status and a synthesized 408 for timeouts, so a caller that wants "did the provider answer" must use `upstream_response()` instead. Do not add more meanings to an existing predicate, add a new one

## Equality

`Arc<Repr>` errors compare with `Arc::ptr_eq`, so a clone equals its original and two separately built errors never compare equal. That exists only so a wrapping enum such as `litellm_inference::Error` can keep its `PartialEq` derive. Do not give a new type pointer equality for any other reason, and do not rely on it in tests. Assert predicates instead, see [testing](testing.md)
