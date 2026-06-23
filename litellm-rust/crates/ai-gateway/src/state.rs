use std::sync::Arc;

use litellm_core::router::Router;

/// Shared application state handed to every route handler.
#[derive(Clone)]
pub struct AppState {
    pub router: Arc<Router>,
    /// Shared secret required as a bearer token on `/v1/realtime`. `None` means
    /// no key is configured and the realtime route fails closed (rejects all
    /// requests). Full per-key auth + budgets are delegated to Python later.
    pub gateway_key: Option<Arc<str>>,
}
