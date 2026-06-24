//! The auth **swap seam**.
//!
//! [`KeyAuthenticator`] is the single trait the extractor depends on. v0 ships
//! [`python::PythonAuthClient`], which delegates verification to the Python proxy
//! over HTTP. A later phase can implement `KeyAuthenticator` natively in Rust (DB
//! lookup, budget checks, etc.) and swap it in at startup — no route, extractor,
//! or state-shape change required, because everything depends on the trait object,
//! not the concrete client.

pub mod python;

use crate::auth::UserApiKeyAuth;

/// Why a key verification failed. `Unauthorized` maps to a `401` for the caller;
/// `Upstream` carries a sanitized backend-error detail (also surfaced as `401` by
/// the extractor so an unreachable backend never silently lets a request through).
#[derive(Debug)]
pub enum AuthError {
    Unauthorized,
    Upstream(String),
}

/// Resolves a raw API key into a [`UserApiKeyAuth`]. The one interface auth
/// backends implement; see the module docs for the swap rationale.
#[axum::async_trait]
pub trait KeyAuthenticator: Send + Sync {
    async fn verify(&self, key: &str) -> Result<UserApiKeyAuth, AuthError>;
}
