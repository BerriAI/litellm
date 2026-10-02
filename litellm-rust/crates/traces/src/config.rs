use litellm_storage_clickhouse::{Error, Storage};

#[derive(Clone)]
pub struct Config {
    storage: Storage,
    retention_days: u32,
}

impl Config {
    pub fn new(database: String, url: &str, retention_days: u32) -> Result<Self, Error> {
        crate::schema_statements(&database, retention_days)?;
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
