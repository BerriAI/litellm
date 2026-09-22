use std::path::Path;

use litellm_cache::Error;

#[derive(Clone, Debug, PartialEq)]
pub enum StoredValue {
    Bytes(Vec<u8>),
    Text(String),
    Integer(i64),
    Float(f64),
    Pickle(Vec<u8>),
}

pub trait DiskStore: Send + Sync + 'static {
    fn directory(&self) -> &Path;
    fn get(&self, key: &str, now: f64) -> Result<Option<StoredValue>, Error>;
    fn set(
        &self,
        key: &str,
        value: StoredValue,
        expire_time: Option<f64>,
        now: f64,
    ) -> Result<(), Error>;
    fn pop(&self, key: &str, now: f64) -> Result<Option<StoredValue>, Error>;
    fn clear(&self) -> Result<(), Error>;
    fn update(
        &self,
        key: &str,
        now: f64,
        apply: &mut dyn FnMut(Option<StoredValue>) -> Result<(StoredValue, Option<f64>), Error>,
    ) -> Result<(), Error>;
}
