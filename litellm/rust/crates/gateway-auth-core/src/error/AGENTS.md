# Errors

`error.rs` is the crate's one `Error`. Each credential family's denials live in `error/<family>.rs`, today `key.rs` and `jwt.rs`, whichever stage raises them. A new family, such as OAuth2 introspection once it moves to Rust, gets its own file with the same shape

## `error/<family>.rs`

- Every denial a family can produce lives in its file, as one top-level `<Family>Denial` enum (`KeyDenial`, `JwtDenial`). A stage module never declares its own error type
- A group of related denials can be its own enum in the same file (`VerifyDenial`, `RbacDenial`), nested in the family enum as an `#[error(transparent)]` `#[from]` variant. When a stage returns the nested enum directly, `crate::Error` gets a `From` for it that wraps through the family variant, so `?` works without spelling the path
- Every message is in `#[error(...)]`, built from the variant's fields. Constants and formatting helpers a message needs stay private to the same file
- A variant carries the values its message and its public mapping need, typed, never a pre-rendered string
- `crate::Error` has one `#[from]` variant per family and never lists a family's denials itself. Denials that no single family owns (budgets, routes, licenses, an unavailable dependency) stay there
- `public.rs` gives each of those enums a `public` method, and a nested one is reached through its family's. The matches are exhaustive, so a new variant without a status does not compile
- Keep each message's opening sentence stable. Clients match on it, so a change goes through the parity tests in `public.rs` as a deliberate update
