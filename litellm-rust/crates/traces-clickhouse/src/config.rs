use std::sync::Arc;

use crate::Error;
use litellm_pagination::KeyRing;
use litellm_storage_clickhouse::Storage;

#[derive(Clone)]
pub struct Config {
    storage: Storage,
    retention_days: u32,
    max_attribute_value_bytes: usize,
    cursor_keys: Arc<KeyRing>,
}

impl Config {
    pub fn new(
        database: String,
        url: &str,
        retention_days: u32,
        max_attribute_value_bytes: usize,
        cursor_keys: Vec<String>,
    ) -> Result<Self, Error> {
        super::schema_statements(&database, retention_days)?;
        Ok(Self {
            storage: Storage::new(database, url)?,
            retention_days,
            max_attribute_value_bytes,
            cursor_keys: Arc::new(KeyRing::new(cursor_keys)?),
        })
    }

    pub fn storage(&self) -> &Storage {
        &self.storage
    }

    pub fn retention_days(&self) -> u32 {
        self.retention_days
    }

    /// Stored span attribute and payload values longer than this are truncated with a marker.
    pub fn max_attribute_value_bytes(&self) -> usize {
        self.max_attribute_value_bytes
    }

    /// Keys that sign read cursors; the first signs and every key verifies.
    pub fn cursor_keys(&self) -> &KeyRing {
        &self.cursor_keys
    }
}
