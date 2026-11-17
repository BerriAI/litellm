use std::time::Duration;

use litellm_python_compat::Value;

use crate::{DecodeError, codec::python_value, dual_cache::MemorySettings};

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
    pub(super) fn memory(self) -> MemorySettings {
        MemorySettings {
            entries: Some(self.memory_entries),
            ttl: self.memory_ttl,
        }
    }
}

pub(super) fn decode(raw: Option<Vec<u8>>) -> Result<Option<Value>, DecodeError> {
    raw.as_deref().map(python_value).transpose()
}
