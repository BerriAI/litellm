//! Finds secrets in text nobody typed as secret, such as an error message, a log line or a logged request, and
//! replaces them. `SecretRedactor` matches secret shapes and `key=value` assignments. `TokenRedactor` matches
//! tokens that start with a known vendor prefix.

mod secret;
mod token;

pub use secret::{REDACTED, SecretRedactor};
pub use token::TokenRedactor;
