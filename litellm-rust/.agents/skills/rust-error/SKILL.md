---
name: rust-error
description: Define, wrap or refactor Rust error types in litellm-rust with thiserror, keeping messages in one place, causes typed, and classification done once
---

# Rust errors

Use this skill when adding an error type or variant, mapping one crate's error into another, or replacing call sites that build errors by hand. `crates/http/src/error.rs` is the reference implementation of everything below

## Where errors live

A crate's errors live in `src/error.rs`, defined with `thiserror`, and are re-exported from `lib.rs`. Default to one top-level `Error` per crate with one variant per failure mode. A failure mode is something a caller handles differently, such as phase, status code, retry, or a message Python parity pins exactly. Failures no caller tells apart share one variant

Keep shared error types minimal and provider-neutral. Provider names, credential types, configuration fields and setup guidance belong in caller-supplied data such as `MissingApiBase { provider: "Azure", guidance: "..." }`, not in dedicated variants or hardcoded shared messages. An exact parity message does not justify a provider-specific variant when caller-supplied context can preserve it

Split into separate types only when different functions fail in disjoint ways that different callers see, so no caller has to match variants its function can never return. Name a split type after what went wrong, not after the function that returns it. A type whose fields are already typed data with the message in `#[error(...)]`, such as `litellm_http::request::HeaderError`, does not need folding into the crate error

## Messages belong to the variant

Every message template is written once, in the variant's `#[error(...)]`. A caller passes the typed facts the template needs and nothing else. None of these may appear at a call site

```rust
Error::Network("WebSocket handshake returned a wrong Sec-WebSocket-Accept".into())
Error::Request(format!("bad header {name}"))
Error::Signature(error.to_string())
```

When a check fails for a distinct reason, that reason is a variant (`Repr::HandshakeAccept`), not a string inside a generic one. Format arguments can call functions and reach into fields (`#[error("could not read {}: {cause}", path.display())]`), so a hand-written `Display` is rarely needed

## Keep causes typed

Store the lower-level error itself, never its string. Flattening a cause loses `source()`, loses its type for downcasting, and for `reqwest::Error` leaks the request URL, which can carry credentials in the path or query. Strip the URL with `without_url()` where a `reqwest::Error` enters the type

Render each cause once. Either the variant's message includes it or `source()` returns it, never both, or a reporter that walks the chain prints it twice. Our boundaries (the gateway error body, the Python bridge) stringify with `to_string()`, so a leaf variant shows its cause in its message and names the field `cause`. A field named `source` becomes the source automatically, and `#[from]` implies `#[source]`. A variant that only wraps another of our errors uses `#[error(transparent)]` so it adds no text

## One conversion per foreign error

Each foreign error type gets exactly one `From` impl in the crate that owns the wrapper, and that impl does all classification and sanitizing. `From<reqwest::Error> for litellm_http::Error` decides connect versus network versus timeout and strips the URL, so every call site is `?` or `.map_err(litellm_http::Error::from)`. A call site never re-derives the classification with `is_connect()` checks or builds a variant from `error.to_string()`

The same holds one layer up. When a lower error needs reclassifying for every caller, do it in the shared type's `From`, not per route. `From<litellm_http::Error> for litellm_inference::Error` turns a header the signer computes into `Unsupported`, so all four routes classify that input the same way

## Nest, don't mirror

Wrap a lower crate's error in one variant and delegate to it. Never re-declare its variants in the upper type with an upper-layer name and a 1:1 mapping function, as OCR once did for every media download failure. The upper type's predicates call through, as in `Self::Http(error) => error.is_request()`

## Ask, don't match

Downstream crates ask a lower error questions instead of matching its nested variants. `litellm_http::Error` answers `dispatched()`, `is_request()`, `is_response()`, `status()`, `upstream_response()` and `tls_source()`. Add a predicate when a real caller needs a new distinction, and name it after what the caller decides, not after the variant

## Opaque errors

Use the opaque newtype only when a crate wraps third-party causes that are not `Clone` or `PartialEq` and a consumer needs those traits, or when the variants must not become public API. `reqwest::Error`, `std::io::Error` and `serde_json::Error` are built this way

```rust
#[derive(Clone, Debug, thiserror::Error)]
#[error(transparent)]
pub struct Error(Arc<Repr>);

#[derive(Debug, thiserror::Error)]
pub(crate) enum Repr {
    #[error("upstream network error: {}", Chain(.0))]
    Network(reqwest::Error),
    #[error("upstream request failed with status {status}: {body}")]
    Status { status: u16, body: String },
}
```

`Repr` holds the real causes. `Arc` makes `Clone` a refcount bump, so `Repr` never has to be `Clone`. `#[error(transparent)]` forwards `Display` and `source()`. Other crates cannot name a variant, so they cannot build an error with a message of their own, and the compiler enforces the rules above. Public constructors exist only for errors another crate genuinely produces, take typed data (`Error::upstream(status, body)`, `Error::computed_header(name)`, `Error::signature(cause)`), and never take a message. `PartialEq` compares by `Arc::ptr_eq`, the same as `SecretError` and `ErrorSource`, so a wrapping enum can keep its derives

A crate whose errors are its own data, which nobody outside matches on, keeps a plain `#[derive(Clone, Debug, thiserror::Error)]` enum

## Testing errors

Assert behavior through the predicates and accessors (`error.upstream_response() == Some((429, "slow down"))`, `!error.dispatched()`), not by constructing an expected variant with a guessed message. Produce errors the way production does, through a public constructor, a real failing call, or a real cause such as `tungstenite::Error::ConnectionClosed`. Inside the owning crate a test may match the private representation through a `#[cfg(test)]` accessor

## thiserror reference

The workspace used thiserror 2.0 when this was written. The [derive docs](https://docs.rs/thiserror/latest/thiserror/) cover format arguments, `#[from]`, `#[source]` and `#[error(transparent)]`. `#[from]` only works on a variant with no other fields besides the source and an optional backtrace, so a variant that needs context such as a path takes its cause through an explicit `map_err`. `#[backtrace]` needs a nightly compiler and is not used here
