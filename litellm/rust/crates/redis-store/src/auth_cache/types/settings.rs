use std::time::Duration;

use crate::dual_cache::MemorySettings;

/// `UserApiKeyCache`'s memory configuration: each partition's size and `default_in_memory_ttl`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Settings {
    pub memory_entries: usize,
    pub memory_ttl: Option<Duration>,
}

impl Default for Settings {
    /// `InMemoryCache`'s 200 entries.
    fn default() -> Self {
        Self {
            memory_entries: 200,
            memory_ttl: None,
        }
    }
}

impl Settings {
    pub(in crate::auth_cache) fn memory(self) -> MemorySettings {
        MemorySettings {
            entries: Some(self.memory_entries),
            ttl: self.memory_ttl,
        }
    }
}
