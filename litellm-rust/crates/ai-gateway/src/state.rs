use std::sync::Arc;

use litellm_router::Router;

/// Shared application state handed to every route handler.
#[derive(Clone)]
pub struct AppState {
    pub router: Arc<Router>,
}
