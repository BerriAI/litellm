# Messages and causes

## Messages belong to the variant

Every message template is written once, in the variant's `#[error(...)]`. A caller passes the typed facts the template needs and nothing else

```rust
// none of these may appear at a call site
Error::Network("WebSocket handshake returned a wrong Sec-WebSocket-Accept".into())
Error::Request(format!("bad header {name}"))
Error::Signature(error.to_string())
Error::AwsSignableRequest(format!("{name} is not visible ASCII"))

// the call site names what happened, the variant owns the words
Repr::HandshakeAccept.into()
Error::signature(error)
Error::NonAsciiHeader { name }
```

When a check fails for a distinct reason, that reason is a variant or a field, never a string inside a generic variant

Format arguments can call functions and reach into fields, so a hand-written `Display` is rarely needed

```rust
#[error("could not read {}: {cause}", path.display())]
#[error("invalid model: {provider} has no model {model:?} - use one of: {}", supported.join(", "))]
```

Keep messages short and lowercase unless Python parity pins the exact text. Do not end them with a period, since a reporter may append `: {cause}`

## Keep causes typed

Store the lower-level error itself, never its string

| Flattening loses | Because |
|---|---|
| `source()` | a `String` has no cause |
| downcasting | the type is gone |
| redaction | `reqwest::Error`'s `Display` includes the request URL, which can carry credentials in the path or query |

If the cause is not `Clone` and the error must be, wrap the cause in `Arc` or make the crate's error opaque. Do not stringify to get `Clone`

```rust
// loses the type and leaks the URL
.map_err(|error| Error::Network(error.to_string()))

// keeps the cause, the From impl strips the URL
.map_err(litellm_http::Error::from)
```

## Strip reqwest URLs on entry

Call `without_url()` where a `reqwest::Error` enters our type, which in practice is `From<reqwest::Error> for litellm_http::Error`. Once it is inside, every later `Display`, `Debug` or chain walk is safe from that URL

`without_url()` only removes the URL reqwest attached. It does not scrub headers, response bodies, provider messages, file paths or anything a variant stores next to it. Those need their own policy, see [boundaries](boundaries.md)

## Render each cause once

Either the variant's message includes the cause or `source()` returns it, never both, or a reporter that walks the chain prints it twice. This is the rule from the [`std::error::Error` docs](https://doc.rust-lang.org/std/error/trait.Error.html)

Our boundaries (the gateway error body and the Python bridge) render with `to_string()` and never walk the chain. So in this workspace a leaf variant puts its cause in its message, and stores it in a field named `cause` so thiserror does not also expose it as the source

| Field | thiserror does | Use for |
|---|---|---|
| `cause: io::Error` rendered in `#[error]` | nothing, `source()` stays `None` | leaf foreign causes, the default here |
| `source: io::Error` or `#[source]` | exposes it as `source()` | causes the message does not mention |
| `#[from] io::Error` | implies `#[source]` and adds `From` | context-free wrapping, message must not mention it |
| `#[error(transparent)]` | forwards both `Display` and `source()` to the inner value | wrapping another of our errors that adds no text |

```rust
// prints the reqwest error twice under a chain-walking reporter
#[error("request failed: {source}")]
Request { source: reqwest::Error },

// leaf, rendered once in the message
#[error("could not read {}: {cause}", path.display())]
Read { path: PathBuf, cause: io::Error },

// nested, adds no text of its own
#[error(transparent)]
Http(litellm_http::Error),
```

`#[error(transparent)]` is an invisible layer, not an extra chain node. A transparent wrapper's `source()` is the inner value's `source()`, so a transparent wrapper over a leaf has no source at all

`reqwest::Error` keeps the useful part of its message in its own source chain (the hyper and io errors), so the `Chain` helper in `crates/http/src/error.rs` renders `reqwest::Error` plus its sources into one message. Use it for any reqwest cause in a message

## #[from] limits

`#[from]` only works on a variant whose only fields are the source and an optional backtrace. A variant that needs context such as a path, a phase or a `TlsSource` takes its cause through an explicit `map_err` or a named constructor

```rust
std::fs::read(&path).map_err(|cause| Repr::Read {
    path: path.clone(),
    tls_source: TlsSource::CaBundle,
    cause,
})?;
```

`#[backtrace]` and automatic `provide()` need a nightly compiler and are not used here. `std::backtrace::Backtrace` itself is stable, but do not add a field of that type to a derived error without checking how thiserror treats it
