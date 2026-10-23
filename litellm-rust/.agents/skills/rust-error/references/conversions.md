# Conversions

## One From per source and target pair

A `From<Foreign> for Ours` exists when the conversion is obvious, infallible and needs no context from the call site. It does all classification and sanitizing for that pair, so every call site is `?` or `.map_err(Ours::from)`

```rust
impl From<reqwest::Error> for Error {
    fn from(error: reqwest::Error) -> Self {
        let error = error.without_url();
        if error.is_timeout() {
            Repr::Timeout(error).into()
        } else if error.is_connect() || error.is_builder() {
            Repr::Connect(error).into()
        } else {
            Repr::Network(error).into()
        }
    }
}
```

A call site never re-derives that classification

```rust
// every route used to repeat this, and each drifted a little
.map_err(|err| {
    if err.is_connect() || err.is_builder() {
        Error::Transport(Connect(err.to_string()))
    } else {
        Error::Transport(Network(err.to_string()))
    }
})?

// now
.map_err(litellm_http::Error::from)?
```

The same foreign type may convert into several of our types, one `From` each. What is not allowed is two places converting the same pair differently

## When the call site knows something the From cannot

One `io::Error` can mean "reading the CA bundle failed" or "resolving the media host failed". A global `From<io::Error>` cannot know which, so there is none. Name the operation at the call site instead

```rust
.map_err(Repr::Resolve)?
.map_err(|cause| Repr::Read { path, tls_source, cause })?
```

Do not drop meaningful context just to use `?`

## Reclassify once, in the shared From

When a lower error needs a different meaning for every caller, do it in the upper type's `From`, not per route

```rust
impl From<litellm_http::Error> for Error {
    fn from(error: litellm_http::Error) -> Self {
        if error.is_computed_header() {
            return Self::Unsupported("request forwards a header AWS SigV4 computes");
        }
        Self::Http(error)
    }
}
```

All four inference routes classify a header the signer computes the same way, because none of them does it themselves

## Nest, translate or add context, never mirror

| The upper layer | Do | Example |
|---|---|---|
| has nothing to add | nest in one transparent variant and delegate predicates | `Self::Http(error) => error.is_request()` |
| changes what the failure means | translate to an upper variant, keep the cause attached when it helps | computed header -> `Unsupported` |
| knows something the lower one does not | a variant with the extra fields plus the cause | `Read { path, tls_source, cause }` |

Never re-declare a lower crate's variants in the upper type with an upper-layer name and a 1:1 mapping function. OCR once did that for every media download failure, `MediaError::BlockedUrl => Error::BlockedDocumentUrl` and so on, which doubled every change and let the two lists drift

## Third-party and host faults

Converting a fault from a crate we do not own into a string variant is still flattening. Prefer a variant that carries the fault, and when an existing `ErrorDetail` constructor fits, use it

```rust
// flattens
.map_err(|error| Error::InvalidRequest(error.to_string().into()))

// keeps the cause typed
.map_err(|error| Error::InvalidRequest(ErrorDetail::invalid("header", error)))
```
