# Testing errors

Assert behavior through predicates and accessors, on errors produced the way production produces them

```rust
// guesses the message and the variant, and cannot be written once the type is opaque
assert_eq!(error, Error::Transport(TransportError::Http { status: 429, body: "slow down".into() }));

// asserts what callers rely on
assert_eq!(error.upstream_response(), Some((429, "slow down")));
assert!(error.dispatched());
assert!(!error.is_request());
```

## Producing the error

In order of preference

1. A real failing call. Bind a `TcpListener`, drop it, then connect to its address for a connect failure. Accept and never answer for a timeout
2. A real cause, such as `tungstenite::Error::ConnectionClosed` or a `serde_json::from_str("{")` error, passed through our `From`
3. A public constructor, such as `Error::upstream(500, "boom".into())`

Inside the owning crate a test may match the private representation through a `#[cfg(test)] fn repr(&self) -> &Repr` accessor, as `crates/http/src/media.rs` does. A public `ErrorDetail` can be destructured anywhere, and its `cause` derefs to `dyn Error` for a `downcast_ref`

## What to cover

| Property | Test shape |
|---|---|
| classification | one `#[case]` per input class, asserting the predicate |
| redaction | build from a URL holding `api_key=secret`, assert the message lacks `secret` and the path |
| cause kept | destructure the variant and downcast the `cause` field, or assert the message contains the root cause's text |
| rendered once | the cause's text appears once in `to_string()` and `source()` is `None`, since the message already carries it |
| boundary mapping | the gateway status and body, or the Python exception type and its `status` attribute |

Assert exact message text only where it is a contract, such as a message Python parity pins. Everywhere else assert that the message contains the facts it must carry
