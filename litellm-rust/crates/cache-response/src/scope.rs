use std::{sync::Arc, time::Duration};

use litellm_cache::{Error, ExactCacheContext};

use crate::{CacheAccess, CacheKeyInput, ResponseCacheRequest, ResponseCacheService};

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CacheScope {
    Shared,
    Isolated(String),
}

#[derive(Clone, Copy, Default)]
pub struct CachePolicy {
    pub caching: Option<bool>,
    pub no_cache: bool,
    pub no_store: bool,
    pub ttl: Option<Duration>,
    pub max_age: Option<Duration>,
}

impl CachePolicy {
    pub fn access(&self) -> CacheAccess {
        let active = self.caching != Some(false);
        CacheAccess {
            reads: active && !self.no_cache,
            writes: active && !self.no_store,
        }
    }

    pub fn enabled(&self) -> bool {
        self.access() != CacheAccess::NONE
    }
}

#[derive(Clone)]
pub struct CacheOptions {
    pub policy: CachePolicy,
    pub scope: CacheScope,
}

impl CacheOptions {
    pub fn new(scope: CacheScope) -> Self {
        Self {
            policy: CachePolicy::default(),
            scope,
        }
    }

    pub fn request(self, surface: &str, input: CacheKeyInput) -> ResponseCacheRequest {
        ResponseCacheRequest {
            key: input,
            surface: surface.to_owned(),
            scope: self.scope,
            access: self.policy.access(),
            context: ExactCacheContext {
                ttl: self.policy.ttl,
            },
            max_age: self.policy.max_age,
        }
    }
}

#[derive(Clone)]
pub struct ScopedCache {
    service: Arc<dyn ResponseCacheService>,
    scope: CacheScope,
}

impl ScopedCache {
    pub fn new(service: Arc<dyn ResponseCacheService>, scope: CacheScope) -> Result<Self, Error> {
        if matches!(scope, CacheScope::Isolated(_)) && !service.config().supports_isolated_scope {
            return Err(Error::UnsupportedOperation);
        }
        Ok(Self { service, scope })
    }

    pub fn shared(service: Arc<dyn ResponseCacheService>) -> Self {
        Self {
            service,
            scope: CacheScope::Shared,
        }
    }

    pub fn service(&self) -> &Arc<dyn ResponseCacheService> {
        &self.service
    }

    pub fn options(&self, policy: Option<CachePolicy>) -> CacheOptions {
        CacheOptions {
            policy: policy.unwrap_or_default(),
            scope: self.scope.clone(),
        }
    }
}
