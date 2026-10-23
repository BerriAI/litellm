# Error shape

## One top-level Error per crate

A crate's errors live in `src/error.rs` and are re-exported from `lib.rs` as `Error`. A caller of any public function in the crate should only ever see `Result<_, crate::Error>`, so one `?` and one set of predicates cover the whole crate

That one type takes one of two forms

| Form | Use when | Example |
|---|---|---|
| Public enum | Callers outside the crate match variants, and every variant's data is ours, cheap to clone and compare | `litellm_inference::Error`, `litellm_auth::Error` |
| Opaque struct over a private `Repr` | The variants hold third-party causes, or the variants must be free to change without breaking callers | `litellm_http::Error` |

Both forms are still one top-level `Error`. The opaque form just moves the enum behind `Repr`, where it is `pub(crate)`

```rust
// public enum
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum Error {
    #[error("missing required field: {0}")]
    MissingField(&'static str),
    #[error(transparent)]
    Http(litellm_http::Error),
}

// opaque struct
#[derive(Clone, Debug, thiserror::Error)]
#[error(transparent)]
pub struct Error(Arc<Repr>);

#[derive(Debug, thiserror::Error)]
pub(crate) enum Repr {
    #[error("could not reach the provider: {}", Chain(.0))]
    Connect(reqwest::Error),
    #[error("upstream request failed with status {status}: {body}")]
    Status { status: u16, body: String },
}
```

Start with the public enum. Move to opaque when a variant needs a non-`Clone` third-party cause, or when downstream crates keep matching variants in ways that duplicate classification. `reqwest::Error`, `std::io::Error` and `serde_json::Error` are built the opaque way

In the opaque form other crates cannot name a variant, so they cannot build an error with their own message. Public constructors exist only for errors another crate genuinely produces, take typed data and never a message, as in `Error::upstream(status, body)`, `Error::computed_header(name)` and `Error::signature(cause)`

`Arc` makes cloning a refcount bump, so `Repr` never has to be `Clone`. `Box<Repr>` is enough when the goal is only a small `Result`. `Arc` is for when callers clone

## What else may sit next to Error

These are not a second top-level error and need no folding in

- Leaf data types whose fields are typed and whose message is in `#[error(...)]`, such as `litellm_http::request::HeaderError`. A wrapping crate nests it as one variant
- Plain data used by predicates, such as `litellm_http::TlsSource`
- A private fault enum one variant carries, such as `PemFault` inside `Repr::InvalidPem`

## Variants are failure modes

A public variant is something a caller handles differently: a phase, a status code, retry versus no retry, fault on the request versus the environment, or a message Python parity pins exactly. Failures no caller tells apart share one variant and differ in their data

```rust
// one variant per provider, so every new provider adds a variant and a match arm
#[error("Missing Azure API Base - set AZURE_API_BASE")]
MissingAzureApiBase,
#[error("Missing Vertex API Base - set VERTEXAI_API_BASE")]
MissingVertexApiBase,

// one failure mode, the provider and its guidance are data
#[error("Missing {provider} API Base - {guidance}")]
MissingApiBase { provider: &'static str, guidance: &'static str },
```

An exact parity message does not justify a provider-specific variant when caller-supplied data can produce the same text

Inside a private `Repr` a variant costs no API, so one per distinct reason is fine when each one carries a different message. When several only differ in a word and share every predicate answer, prefer one variant with a fieldless reason enum, so predicates list one arm instead of five

```rust
#[error("WebSocket handshake returned {0}")]
Handshake(HandshakeFault),

#[derive(Debug, thiserror::Error)]
pub(crate) enum HandshakeFault {
    #[error("a wrong Sec-WebSocket-Accept")]
    Accept,
    #[error("an invalid Upgrade header")]
    Upgrade,
}
```

## When to split

Split into a separate type only when different functions fail in disjoint ways and different callers see them, so no caller matches variants its function can never return. A type per function overdoes it

Name a split type after what went wrong, not after the function that returns it. `ConnectError` reads better than `OpenSessionError`, and a unit struct is fine for a single failure mode

A split type still converts into the crate's `Error` when a caller further up needs one `Result` type
