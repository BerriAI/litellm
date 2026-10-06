use std::time::Duration;

use serde::Deserialize;

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

#[derive(Clone, Copy, Debug)]
pub struct CachePolicy {
    pub access: CacheAccess,
    pub ttl: Option<Duration>,
    pub max_age: Option<Duration>,
}

impl Default for CachePolicy {
    fn default() -> Self {
        Self {
            access: CacheAccess::READ_WRITE,
            ttl: None,
            max_age: None,
        }
    }
}

impl CachePolicy {
    pub fn enabled(&self) -> bool {
        self.access != CacheAccess::NONE
    }
}
