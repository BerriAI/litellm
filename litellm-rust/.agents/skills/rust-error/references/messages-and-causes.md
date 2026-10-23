# Messages and causes

## Messages belong to the variant

Every message template is written once, in the variant's `#[error(...)]`. A caller passes the typed facts the template needs and nothing else

```rust
// none of these may appear at a call site
Error::Network("WebSocket handshake returned a wrong Sec-WebSocket-Accept".into())
Error::Request(format!("bad header {name}"))
Error::Signature(error.to_string())

// the call site names what happened, the variant owns the words
Repr::Handshake(HandshakeFault::Accept).into()
Error::signature(error)
Error::NonAsciiHeader { name }
```

When a check fails for a distinct reason, that reason is a variant or a field, never a string inside a generic variant. A `Message(String)` variant, or a `From<String>` / `From<&str>` into an error, is the hole every stringly call site goes through, so neither exists

Format arguments can call functions and reach into fields, so a hand-written `Display` is rarely needed

```rust
#[error("could not read {}", path.display())]
#[error("invalid model: {provider} has no model {model:?} - use one of: {}", supported.join(", "))]
```

Keep messages short and lowercase, with no trailing period, unless Python parity pins the exact text. A reporter appends `: {cause}` after them. Never format `Debug` (`{:?}`) of an error, or of a type with no `Display`, into a message

## The cause is the source, never the message

This is the rule from the [`std::error::Error` docs](https://doc.rust-lang.org/std/error/trait.Error.html): an error's `Display` describes that layer only, and the error underneath is returned by `source()`. thiserror wires `source()` from a field marked `#[source]`, a `#[from]` field, or a field named `source`

```rust
// prints the cause twice under any reporter
#[error("could not read {}: {source}", path.display())]
Read { path: PathBuf, source: io::Error },
#[error("invalid JSON: {0}")]
Json(#[from] serde_json::Error),

// this layer's words, the cause is the source
#[error("could not read {}", path.display())]
Read { path: PathBuf, source: io::Error },
#[error("invalid JSON")]
Json(#[from] serde_json::Error),
```

| Field | thiserror does | Use for |
|---|---|---|
| `source: E` or `#[source] cause: E` | exposes it as `source()` | a cause with context fields next to it |
| `#[from] E` | implies `#[source]` and adds `From` | context-free wrapping, see [conversions](conversions.md) |
| `#[error(transparent)]` | forwards both `Display` and `source()` to the inner value | nesting another of our errors that adds no text |
| no attribute, any other name | nothing, the field is plain data | facts the message renders, such as a path or a status |

`#[error(transparent)]` is an invisible layer, not an extra chain node, so a transparent wrapper's `source()` is the inner value's `source()`

A field that holds an error but is neither rendered nor a source hides that error from every reader. Every error-typed field is a source. A fieldless reason enum, such as `HandshakeFault` or `PagesFault`, is data and not a cause, so the variant renders it in its message

## Keep causes typed

Store the lower-level error itself, never its string, and never drop it with `map_err(|_| ...)`

| Flattening or dropping loses | Because |
|---|---|
| the chain | a `String` has no `source()` |
| downcasting | the type is gone |
| redaction | `reqwest::Error`'s `Display` includes the request URL, which can carry credentials in the path or query |

If the cause is not `Clone` and the error must be, hold it as `#[source] litellm_error::Source`, a cloneable `Arc<dyn Error + Send + Sync>` that is transparent to `Display` and `source()`, or make the crate's error opaque. `litellm_error::find::<T>(&error)` walks the chain and looks through `Source`, so it is how anyone reaches a typed cause. A third-party `Box<dyn Error + Send + Sync>` converts with `Source::from`, and an `anyhow::Error` a dependency returns goes through `Box` first, `Source::from(Box::<dyn Error + Send + Sync>::from(error))`, and a dependency's own boxed alias such as `tokenizers::Error` is held as it is. A source boxed to keep the variant small is a `Box<E>` node of the chain, so it is reached as `find::<Box<E>>`. Never put `#[error(transparent)]` on a `Source`, the erased error would then be no node of the chain and `find` could not reach it. Do not stringify or drop a cause to get `Clone` or `PartialEq`

```rust
// loses the type and leaks the URL
.map_err(|error| Error::Network(error.to_string()))
// loses everything
.map_err(|_| Error::Transport)

// keeps the cause, the From impl strips the URL
.map_err(litellm_http::Error::from)
```

A cause that must not be shown at all, such as a secret manager's response, is still a source. The type derives [`veil::Redact`](https://docs.rs/veil) with `#[redact]` on the field, so `Debug` hides it, and the boundary decides how much of the chain to show, see [boundaries](boundaries.md)

## Strip reqwest URLs on entry

Every `reqwest::Error` enters our types through `From<reqwest::Error> for litellm_http::Error`, which calls `without_url()`. A crate that talks HTTP holds `litellm_http::Error`, never `reqwest::Error`, so every later `Display`, `Debug` or chain walk is safe from that URL

`without_url()` only removes the URL reqwest attached. It does not scrub headers, response bodies, provider messages, file paths or anything a variant stores next to it. Those need their own policy, see [boundaries](boundaries.md)

## Rendering

Anything that shows an error to someone else renders the whole chain with `litellm_error::Report`, which writes each layer once joined by `: `, the single-line form of the unstable [`std::error::Report`](https://doc.rust-lang.org/std/error/struct.Report.html). `to_string()` of an error is only its top layer and is never what a client, a Python exception or a log line shows

```rust
let message = litellm_error::Report::new(&error).to_string();
// "could not reach the provider: error sending request: client error (Connect): tcp connect error: Connection refused"
```

## #[from] limits

`#[from]` only works on a variant whose only fields are the source and an optional backtrace. A variant that needs context such as a path, a phase or a `TlsSource` takes its cause through an explicit `map_err` or a named constructor

```rust
std::fs::read(&path).map_err(|source| Repr::Read {
    path: path.clone(),
    tls_source: TlsSource::CaBundle,
    source,
})?;
```

`#[backtrace]` and automatic `provide()` need a nightly compiler and are not used here
