mod error;
mod insert;
mod schema;

pub use error::Error;
pub use insert::{encode_rows, insert_rows};
use litellm_storage_clickhouse::Storage;
pub use schema::ensure_schema;

#[derive(Clone)]
pub struct Config {
    storage: Storage,
    retention_days: u32,
}

impl Config {
    pub fn new(database: String, url: &str, retention_days: u32) -> Result<Self, Error> {
        if retention_days == 0 {
            return Err(Error::InvalidSchema);
        }
        Ok(Self {
            storage: Storage::new(database, url)?,
            retention_days,
        })
    }

    pub fn storage(&self) -> &Storage {
        &self.storage
    }

    pub fn retention_days(&self) -> u32 {
        self.retention_days
    }
}
