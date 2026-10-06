use std::time::Duration;

use litellm_cache::{CacheContext, ExactCacheContext};

use crate::{CacheControls, CacheKeyInput};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum RequestRewrite {
    #[default]
    Unchanged,
    Rewritten,
}

impl From<&CacheKeyInput> for RequestRewrite {
    fn from(input: &CacheKeyInput) -> Self {
        match input.rewritten_request {
            Some(_) => Self::Rewritten,
            None => Self::Unchanged,
        }
    }
}

#[derive(Clone)]
pub struct ResponseCacheRequest<C: CacheContext = ExactCacheContext> {
    pub key: CacheKeyInput,
    pub controls: CacheControls,
    pub context: C,
    pub max_age: Option<Duration>,
    pub rewrite: RequestRewrite,
}

impl<C: CacheContext + Default> ResponseCacheRequest<C> {
    pub fn new(key: CacheKeyInput) -> Self {
        Self {
            rewrite: RequestRewrite::from(&key),
            key,
            controls: CacheControls::enabled(),
            context: C::default(),
            max_age: None,
        }
    }
}

impl<C: CacheContext> ResponseCacheRequest<C> {
    pub fn with_context<D: CacheContext>(self, context: D) -> ResponseCacheRequest<D> {
        ResponseCacheRequest {
            key: self.key,
            controls: self.controls,
            context,
            max_age: self.max_age,
            rewrite: self.rewrite,
        }
    }
}
