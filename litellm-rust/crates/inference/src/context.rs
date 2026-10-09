use litellm_cache_response::CachePolicy;
use litellm_host::interceptors::{ExecutionFacts, Interceptors, RawResponse};

use crate::RouteError;

pub struct CallContext<'a, I> {
    pub interceptors: &'a I,
    pub cache: CachePolicy,
}

impl<'a, I: Interceptors<RouteError>> CallContext<'a, I> {
    pub fn new(interceptors: &'a I, cache: Option<CachePolicy>) -> Self {
        Self {
            interceptors,
            cache: cache.unwrap_or_default(),
        }
    }

    pub async fn result_ready(&self, facts: ExecutionFacts) -> Result<(), RouteError> {
        self.interceptors.result_ready(facts).await
    }

    pub async fn response_received(&self, body: &str) -> Result<(), RouteError> {
        self.interceptors
            .after_provider_response(RawResponse {
                body: body.to_owned(),
            })
            .await
            .map_err(RouteError::post_call)
    }
}
