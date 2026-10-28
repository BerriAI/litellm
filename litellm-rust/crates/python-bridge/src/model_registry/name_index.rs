use std::collections::HashMap;

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyString};

/// The one name index of decision 18. A normalized name to the actual key, and the keys each raw `litellm_provider`
/// names. Never a row, since rows are read live, so only adding or removing keys can make it stale
#[derive(Debug, Default)]
pub(crate) struct NameIndex {
    dict_id: usize,
    len: usize,
    by_lower: HashMap<String, String>,
    by_provider: HashMap<String, Vec<String>>,
}

impl NameIndex {
    pub(crate) fn build(dict: &Bound<'_, PyDict>) -> Self {
        let mut by_lower = HashMap::with_capacity(dict.len());
        let mut by_provider: HashMap<String, Vec<String>> = HashMap::new();
        for (key, row) in dict.iter() {
            let Ok(key) = key.cast::<PyString>().map(|key| key.to_string()) else {
                continue;
            };
            let provider = row
                .cast::<PyDict>()
                .ok()
                .and_then(|row| row.get_item("litellm_provider").ok().flatten())
                .and_then(|provider| provider.extract::<String>().ok());
            if let Some(provider) = provider {
                by_provider.entry(provider).or_default().push(key.clone());
            }
            by_lower.insert(key.to_lowercase(), key);
        }
        Self {
            dict_id: dict.as_ptr() as usize,
            len: dict.len(),
            by_lower,
            by_provider,
        }
    }

    pub(crate) fn is_current(&self, dict: &Bound<'_, PyDict>) -> bool {
        self.dict_id == dict.as_ptr() as usize && self.len == dict.len()
    }

    pub(crate) fn by_lower(&self, name: &str) -> Option<&str> {
        self.by_lower.get(&name.to_lowercase()).map(String::as_str)
    }

    pub(crate) fn provider_keys(&self, provider: &str) -> &[String] {
        self.by_provider.get(provider).map_or(&[], Vec::as_slice)
    }
}
