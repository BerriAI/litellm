//! `auth/client` — turn a raw API key into an authenticated identity.
//!
//! When a request presents a virtual key, the gateway needs to know two things:
//! is the key valid, and who does it belong to. This module is the backend that
//! answers that — given a key string, it returns a [`UserApiKeyAuth`] or rejects
//! it. In v0 the answer comes from the Python control plane over HTTP
//! ([`python::PythonAuthClient`]); the [`KeyAuthenticator`] trait lets that be
//! swapped for a native Rust implementation later without touching callers.
//!
//! ## The swap seam
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
    /// Resolve `key` for the given `route` (the gateway's own request path, e.g.
    /// `/v1/realtime`) and the requested `model` (if any). Both are forwarded so
    /// the backend enforces the key/team's route AND model permissions against
    /// what's actually being requested — not just that the key exists.
    async fn verify(
        &self,
        key: &str,
        route: &str,
        model: Option<&str>,
    ) -> Result<UserApiKeyAuth, AuthError>;
}
