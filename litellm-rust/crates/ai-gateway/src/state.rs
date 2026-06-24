use std::sync::Arc;

use crate::io::realtime_pool::RealtimePool;
use litellm_core::router::Router;

/// Shared application state handed to every route handler.
#[derive(Clone)]
pub struct AppState {
    pub router: Arc<Router>,
    /// The gateway master key. Any caller presenting it as a bearer token may
    /// invoke the gateway. `None` → auth not configured (routes fail closed).
    pub master_key: Option<Arc<str>>,
    /// Pre-warmed upstream realtime connection pool. Disabled
    /// (`RealtimePool::disabled()`) when `REALTIME_POOL_SIZE=0`, in which case
    /// every realtime connect fresh-dials exactly as before.
    pub realtime_pool: Arc<RealtimePool>,
    /// Verifies virtual keys. A trait object so the v0 "call Python" backend can be
    /// swapped for a native Rust one at startup without touching routes or state.
    pub authenticator: Arc<dyn crate::auth::KeyAuthenticator>,
    /// Bounded TTL cache of verified keys, in front of `authenticator`.
    pub key_cache: Arc<crate::auth::KeyCache>,
}

impl AppState {
    /// Constant-time check of `token` against the configured master key.
    pub fn is_master_key(&self, token: &str) -> bool {
        crate::auth::user_api_key::is_master_key(self.master_key.as_deref(), token)
    }
}
