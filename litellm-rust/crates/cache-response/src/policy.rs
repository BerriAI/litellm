use std::time::Duration;

use serde::Deserialize;

use crate::CacheScope;

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

#[derive(Clone, Copy, Debug, Default)]
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

#[derive(Clone, Debug)]
pub struct CacheOptions {
    pub policy: CachePolicy,
    pub scope: CacheScope,
}

impl CacheOptions {
    pub fn shared(policy: CachePolicy) -> Self {
        Self {
            policy,
            scope: CacheScope::Shared,
        }
    }
}
