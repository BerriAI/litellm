use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Default, Deserialize, Serialize)]
pub struct CacheControls {
    pub supported_call_type: bool,
    pub configured: bool,
    pub default_on: bool,
    pub caching: Option<bool>,
    pub no_cache: bool,
    pub no_store: bool,
    #[serde(default)]
    pub use_cache: bool,
}

impl CacheControls {
    pub const fn enabled() -> Self {
        Self {
            supported_call_type: true,
            configured: true,
            default_on: true,
            caching: None,
            no_cache: false,
            no_store: false,
            use_cache: false,
        }
    }

    pub fn reads(self) -> bool {
        self.active() && !self.no_cache
    }

    pub fn writes(self) -> bool {
        self.active() && !self.no_store
    }

    fn active(self) -> bool {
        self.supported_call_type
            && self.configured
            && self.caching.unwrap_or(true)
            && (self.default_on || self.use_cache)
    }
}

pub fn should_use_cache(controls: CacheControls) -> bool {
    controls.reads() || controls.writes()
}
