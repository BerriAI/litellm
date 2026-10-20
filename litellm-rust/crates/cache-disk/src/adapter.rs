use litellm_cache::Error;

use crate::StoredValue;

pub trait ValueAdapter: Send + Sync + 'static {
    fn read(&self, value: StoredValue) -> Result<Option<Vec<u8>>, Error>;
    fn write(&self, payload: Vec<u8>) -> StoredValue;
    fn counter_seed(&self, value: Option<StoredValue>) -> Result<f64, Error>;
    fn counter_value(&self, value: f64) -> StoredValue;
}
