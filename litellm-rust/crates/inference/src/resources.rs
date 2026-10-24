use std::sync::Arc;

use litellm_auth::AuthServices;
use litellm_http::HttpClientPool;

#[derive(Clone)]
pub struct CoreResources<C = ()> {
    pub pool: Arc<HttpClientPool>,
    pub auth: Arc<AuthServices>,
    pub catalog: C,
}

impl CoreResources<()> {
    pub fn new(pool: Arc<HttpClientPool>) -> Self {
        Self {
            pool,
            auth: Arc::new(AuthServices::default()),
            catalog: (),
        }
    }
}

impl<C> CoreResources<C> {
    pub fn with_catalog<T>(self, catalog: T) -> CoreResources<T> {
        CoreResources {
            pool: self.pool,
            auth: self.auth,
            catalog,
        }
    }
}
