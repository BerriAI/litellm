---
name: rust-redaction
description: Keep secrets out of Debug output in litellm-rust with SecretValue and veil. Use when a struct or enum holds a password, token, key, credential URL or credential provider, when writing or reviewing a Debug impl, or when a type with secrets gets logged or traced
metadata:
  internal: true
---

# Rust redaction

Every secret in a Debug output is hidden the same way, so a reviewer can tell at a glance that a field is covered and a new field cannot slip through a hand-written impl

## Where SecretValue lives

`SecretValue` lives in `crates/secret-value` and is imported as `litellm_secret_value::SecretValue` everywhere. No crate re-exports it, so there is one import path. Depend on `litellm-secret-value` directly. Do not pull in `litellm-auth-types` or `litellm-auth` just to get it

`crates/secret-value` holds only the type. Credential acquisition (`TokenProvider`, `CredentialPlan`, provider param tables) stays in `litellm-auth-types`, and generic clients such as `litellm-redis-client` must not depend on it

## Rules

- Never hand-write `impl Debug` to hide a value, and never invent markers like `"<redacted>"`, `"<configured>"` or `.map(|_| "...")`. A hand-written impl silently prints any field added later
- A secret that crosses a crate boundary or lives in config is typed as `SecretValue`. It redacts itself, compares in constant time and only gives up the plaintext through `expose()`. A type whose only secrets are `SecretValue` fields derives plain `Debug`
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
- A secret that has to stay a raw type, because a serde wire shape or a third-party struct owns the field, is covered by deriving `veil::Redact` on the holder and marking the field `#[redact(with = "[REDACTED]")]`. `Redact` replaces `Debug`, so do not derive both
- A URL that can carry userinfo is a secret as a whole. Do not parse it to show the host. Keep it as a `SecretValue`
- A bag of unparsed settings that may hold credentials (`additional_fields`, raw kwargs) is redacted as a whole with `#[redact(with = "[REDACTED]")]`
- An opaque handle over `Arc<dyn Trait>` whose trait has no `Debug` gets a newtype whose `Debug` prints only the type name with `finish_non_exhaustive()`. That impl shows nothing, so it is not a redaction impl and the first rule does not apply. A type holding such a handle can derive `Debug`
- Never put a secret in a tracing field or an error message. Log the type with `?value`, or log only non-secret fields. See the rust-tracing and rust-error skills

## Tests

- One test per type that holds a secret, built the way production builds it, asserting the plaintext is absent from `format!("{value:?}")` and `[REDACTED]` is present
- Use a distinctive plaintext (`"credential-value"`) so the absence check cannot pass by accident
