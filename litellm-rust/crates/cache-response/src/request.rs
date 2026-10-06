use std::time::Duration;

use litellm_cache::{CacheContext, ExactCacheContext};
use serde::Deserialize;

use crate::{CacheKeyInput, CacheScope};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CacheAccess {
    pub reads: bool,
    pub writes: bool,
}

impl CacheAccess {
    pub const READ_WRITE: Self = Self {
        reads: true,
        writes: true,
    };
    pub const NONE: Self = Self {
        reads: false,
        writes: false,
    };
}

#[derive(Clone)]
pub struct ResponseCacheRequest<C: CacheContext = ExactCacheContext> {
    pub key: CacheKeyInput,
    pub surface: String,
    pub scope: CacheScope,
    pub access: CacheAccess,
    pub context: C,
    pub max_age: Option<Duration>,
}

impl<C: CacheContext + Default> ResponseCacheRequest<C> {
    pub fn new(key: CacheKeyInput) -> Self {
        Self {
            key,
            surface: String::new(),
            scope: CacheScope::Shared,
            access: CacheAccess::READ_WRITE,
            context: C::default(),
            max_age: None,
        }
    }
}

impl<C: CacheContext> ResponseCacheRequest<C> {
    pub fn with_context<D: CacheContext>(self, context: D) -> ResponseCacheRequest<D> {
        ResponseCacheRequest {
            key: self.key,
            surface: self.surface,
            scope: self.scope,
            access: self.access,
            context,
            max_age: self.max_age,
        }
    }
}
