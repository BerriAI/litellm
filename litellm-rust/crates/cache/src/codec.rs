use std::marker::PhantomData;

use serde::{Serialize, de::DeserializeOwned};

use crate::Error;

pub trait CacheCodec: Send + Sync {
    type Value: Clone + Send + Sync + 'static;

    fn encode(&self, value: &Self::Value) -> Result<Vec<u8>, Error>;

    fn decode(&self, bytes: &[u8]) -> Result<Self::Value, Error>;
}

pub struct JsonCodec<V>(PhantomData<fn() -> V>);

impl<V> Clone for JsonCodec<V> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<V> Copy for JsonCodec<V> {}

impl<V> Default for JsonCodec<V> {
    fn default() -> Self {
        Self::new()
    }
}

impl<V> JsonCodec<V> {
    pub const fn new() -> Self {
        Self(PhantomData)
    }
}

impl<V> CacheCodec for JsonCodec<V>
where
    V: Clone + Send + Sync + Serialize + DeserializeOwned + 'static,
{
    type Value = V;

    fn encode(&self, value: &Self::Value) -> Result<Vec<u8>, Error> {
        serde_json::to_vec(value).map_err(|_| Error::InvalidEntry)
    }

    fn decode(&self, bytes: &[u8]) -> Result<Self::Value, Error> {
        serde_json::from_slice(bytes).map_err(|_| Error::InvalidEntry)
    }
}
