use std::sync::Arc;

use litellm_core::router::Router;

/// Shared application state handed to every route handler.
#[derive(Clone)]
pub struct AppState {
    pub router: Arc<Router>,
    /// The gateway master key. Any caller presenting it as a bearer token may
    /// invoke the gateway. `None` → auth not configured (routes fail closed).
    pub master_key: Option<Arc<str>>,
}
