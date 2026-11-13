---
name: rust-redaction
description: Keep secrets out of Debug output in litellm-rust with SecretValue and Redacted. Use when a struct or enum holds a password, token, key, credential URL or credential provider, when writing or reviewing a Debug impl, or when a type with secrets gets logged or traced
metadata:
  internal: true
---

# Rust redaction

Every secret in a Debug output is hidden the same way, so a reviewer can tell at a glance that a field is covered and a new field cannot slip through a hand-written impl

## Where SecretValue and Redacted live

`SecretValue` and `Redacted<T>` live in `crates/secret-value` and are imported as `litellm_secret_value::{Redacted, SecretValue}` everywhere. No crate re-exports them, so there is one import path. Depend on `litellm-secret-value` directly. Do not pull in `litellm-auth-types` or `litellm-auth` just to get them

These two types are the only redaction tools. veil is not a dependency of the workspace, so do not add it back

`crates/secret-value` holds only the redaction types. Credential acquisition (`TokenProvider`, `CredentialPlan`, provider param tables) stays in `litellm-auth-types`, and generic clients such as `litellm-redis-client` must not depend on it

## Rules

- Never hand-write `impl Debug` to hide a value, and never invent markers like `"<redacted>"`, `"<configured>"` or `.map(|_| "...")`. A hand-written impl silently prints any field added later
- A secret string that crosses a crate boundary or lives in config is typed as `SecretValue`, never `Redacted<String>`. It redacts itself, compares in constant time and only gives up the plaintext through `expose()`. A type whose only secrets are `SecretValue` fields derives plain `Debug`
  ```rust
  #[derive(Clone, Debug, PartialEq)]
  pub struct HostEndpoint {
      pub host: String,
      pub port: u16,
      pub username: Option<SecretValue>,
      pub password: Option<SecretValue>,
      pub tls: bool,
      pub db: i64,
  }
  ```
- Keep a secret as `SecretValue` until the last conversion into the third-party type (redis-rs, reqwest, an SDK) and call `expose()` there. Calling `expose()` early spreads plaintext `String`s through our own types
- Any other value that may hold a secret is typed as `Redacted<T>`: a bag of unparsed settings (`additional_fields`, raw kwargs, callback lists), a body, a hash, a proxy URL, a provider handle (`Redacted<Arc<dyn TokenProvider>>`). It prints only `[REDACTED]`, does not need `T: Debug`, and gives up the value through `expose()` or `into_inner()`. It deserializes and serializes as `T` (`#[serde(transparent)]`, so `#[serde(flatten)]` still works). The holder derives plain `Debug`. In `litellm-config`, `AdditionalFields` is already `Redacted<_>`, so every flattened bag is covered by its type
- An error cause whose `Debug` may carry a secret (an SDK error, a secret manager response) is held as `#[source] Redacted<E>` and built with `Redacted::new`. Only its `Debug` is hidden. Its `Display` and the rest of the chain pass through unchanged, and `litellm_error::find` looks through it the same way it looks through `Source`. Do not put `#[from]` on it, since thiserror would generate `From<Redacted<E>>`. When call sites convert `E` with `?`, write one `impl From<E>` that wraps it
- A URL that can carry userinfo is a secret as a whole. Do not parse it to show the host. Keep it as a `SecretValue`
- An opaque handle over `Arc<dyn Trait>` whose trait has no `Debug` gets a newtype whose `Debug` prints only the type name with `finish_non_exhaustive()`. That impl shows nothing, so it is not a redaction impl and the first rule does not apply. A type holding such a handle can derive `Debug`
- Never put a secret in a tracing field or an error message. Log the type with `?value`, or log only non-secret fields. See the rust-tracing and rust-error skills

## Tests

- One test per type that holds a secret, built the way production builds it, asserting the plaintext is absent from `format!("{value:?}")` and `[REDACTED]` is present
- Use a distinctive plaintext (`"credential-value"`) so the absence check cannot pass by accident
