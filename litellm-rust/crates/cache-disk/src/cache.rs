use std::{
    path::Path,
    sync::Arc,
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use litellm_cache::{
    BaseCache, BatchCache, BatchEntry, CacheCodec, CacheConnectionResult, CacheConnectionStatus,
    CounterCache, DeleteCache, Error, ExactCacheContext, FlushCache,
};
use serde_json::Value;

use crate::{DiskStore, DiskcacheSqliteStore, StoredValue, pickle};

pub struct DiskCache<S, D = DiskcacheSqliteStore> {
    store: Arc<D>,
    codec: S,
}

impl<S: CacheCodec> DiskCache<S> {
    pub fn open(directory: impl AsRef<Path>, codec: S) -> Result<Self, Error> {
        Ok(Self {
            store: Arc::new(DiskcacheSqliteStore::open(directory)?),
            codec,
        })
    }
}

impl<S: CacheCodec, D: DiskStore> DiskCache<S, D> {
    pub fn with_store(store: D, codec: S) -> Self {
        Self {
            store: Arc::new(store),
            codec,
        }
    }

    pub fn directory(&self) -> &Path {
        self.store.directory()
    }

    fn decode_stored(&self, value: StoredValue) -> Result<Option<S::Value>, Error> {
        let Some(bytes) = payload(value)? else {
            return Ok(None);
        };
        self.codec.decode(&bytes).map(Some)
    }

    async fn run_blocking<T, F>(store: Arc<D>, operation: F) -> Result<T, Error>
    where
        T: Send + 'static,
        F: FnOnce(&D) -> Result<T, Error> + Send + 'static,
    {
        tokio::task::spawn_blocking(move || operation(&store))
            .await
            .map_err(|_| Error::Unavailable)?
    }
}

impl<S: CacheCodec, D: DiskStore> BaseCache for DiskCache<S, D> {
    type Value = S::Value;
    type Context = ExactCacheContext;

    fn get_ttl(&self, context: &Self::Context) -> Option<Duration> {
        context.ttl
    }

    fn set_cache(
        &self,
        key: &str,
        value: Self::Value,
        context: &Self::Context,
    ) -> Result<(), Error> {
        let value = StoredValue::Bytes(self.codec.encode(&value)?);
        let expire_time = context.ttl.map(|ttl| unix_now() + ttl.as_secs_f64());
        self.store.set(key, value, expire_time, unix_now())
    }

    fn get_cache(&self, key: &str, _: &Self::Context) -> Result<Option<Self::Value>, Error> {
        self.store
            .get(key, unix_now())?
            .map(|value| self.decode_stored(value))
            .transpose()
            .map(|value| value.flatten())
    }

    async fn async_set_cache(
        &self,
        key: &str,
        value: Self::Value,
        context: ExactCacheContext,
    ) -> Result<(), Error> {
        let value = StoredValue::Bytes(self.codec.encode(&value)?);
        let expire_time = context.ttl.map(|ttl| unix_now() + ttl.as_secs_f64());
        let key = key.to_string();
        Self::run_blocking(Arc::clone(&self.store), move |store| {
            store.set(&key, value, expire_time, unix_now())
        })
        .await
    }

    async fn async_get_cache(
        &self,
        key: &str,
        _: &ExactCacheContext,
    ) -> Result<Option<Self::Value>, Error> {
        let key = key.to_string();
        let value = Self::run_blocking(Arc::clone(&self.store), move |store| {
            store.get(&key, unix_now())
        })
        .await?;
        value
            .map(|value| self.decode_stored(value))
            .transpose()
            .map(|value| value.flatten())
    }

    async fn async_set_cache_pipeline(
        &self,
        entries: Vec<(String, Self::Value)>,
        context: ExactCacheContext,
    ) -> Result<(), Error> {
        let entries = entries
            .into_iter()
            .map(|(key, value)| {
                self.codec
                    .encode(&value)
                    .map(|value| (key, StoredValue::Bytes(value)))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let expire_after = context.ttl;
        Self::run_blocking(Arc::clone(&self.store), move |store| {
            for (key, value) in entries {
                let expire_time = expire_after.map(|ttl| unix_now() + ttl.as_secs_f64());
                store.set(&key, value, expire_time, unix_now())?;
            }
            Ok(())
        })
        .await
    }

    async fn disconnect(&self) -> Result<(), Error> {
        Ok(())
    }

    async fn test_connection(&self) -> Result<CacheConnectionResult, Error> {
        let result = Self::run_blocking(Arc::clone(&self.store), |store| {
            store.probe().map(|_| CacheConnectionResult {
                status: CacheConnectionStatus::Success,
                message: "Disk cache connection test successful".into(),
                error: None,
            })
        })
        .await;
        Ok(match result {
            Ok(result) => result,
            Err(error) => CacheConnectionResult {
                status: CacheConnectionStatus::Failed,
                message: format!("Disk cache connection failed: {error}"),
                error: Some(error.to_string()),
            },
        })
    }
}

impl<S: CacheCodec, D: DiskStore> BatchCache for DiskCache<S, D> {
    fn batch_get_cache(
        &self,
        keys: &[String],
        context: &ExactCacheContext,
    ) -> Result<Vec<BatchEntry<Self::Value>>, Error> {
        keys.iter()
            .map(|key| match self.get_cache(key, context) {
                Ok(Some(value)) => Ok(BatchEntry::Hit(value)),
                Ok(None) => Ok(BatchEntry::Miss),
                Err(Error::InvalidEntry) => Ok(BatchEntry::Invalid),
                Err(error) => Err(error),
            })
            .collect()
    }

    async fn async_batch_get_cache(
        &self,
        keys: Vec<String>,
        _: ExactCacheContext,
    ) -> Result<Vec<BatchEntry<Self::Value>>, Error> {
        let values = Self::run_blocking(Arc::clone(&self.store), move |store| {
            keys.into_iter()
                .map(|key| store.get(&key, unix_now()).map(|value| (key, value)))
                .collect::<Result<Vec<_>, _>>()
        })
        .await?;
        values
            .into_iter()
            .map(|(_, value)| match value {
                None => Ok(BatchEntry::Miss),
                Some(value) => match self.decode_stored(value) {
                    Ok(Some(value)) => Ok(BatchEntry::Hit(value)),
                    Ok(None) => Ok(BatchEntry::Miss),
                    Err(Error::InvalidEntry) => Ok(BatchEntry::Invalid),
                    Err(error) => Err(error),
                },
            })
            .collect()
    }
}

impl<S: CacheCodec, D: DiskStore> DeleteCache for DiskCache<S, D> {
    fn delete_cache(&self, key: &str) -> Result<(), Error> {
        self.store.pop(key, unix_now()).map(|_| ())
    }

    async fn async_delete_cache(&self, key: &str) -> Result<(), Error> {
        let key = key.to_string();
        Self::run_blocking(Arc::clone(&self.store), move |store| {
            store.pop(&key, unix_now()).map(|_| ())
        })
        .await
    }
}

impl<S: CacheCodec, D: DiskStore> FlushCache for DiskCache<S, D> {
    fn flush_cache(&self) -> Result<(), Error> {
        self.store.clear()
    }

    async fn async_flush_cache(&self) -> Result<(), Error> {
        Self::run_blocking(Arc::clone(&self.store), |store| store.clear()).await
    }
}

impl<S: CacheCodec<Value = f64>, D: DiskStore> CounterCache for DiskCache<S, D> {
    fn increment_cache(
        &self,
        key: &str,
        amount: f64,
        context: ExactCacheContext,
    ) -> Result<f64, Error> {
        increment(self.store.as_ref(), key, amount, context.ttl)
    }

    async fn async_increment(
        &self,
        key: &str,
        amount: f64,
        context: ExactCacheContext,
    ) -> Result<f64, Error> {
        let key = key.to_string();
        Self::run_blocking(Arc::clone(&self.store), move |store| {
            increment(store, &key, amount, context.ttl)
        })
        .await
    }
}

fn increment<D: DiskStore>(
    store: &D,
    key: &str,
    amount: f64,
    ttl: Option<Duration>,
) -> Result<f64, Error> {
    let mut result = None;
    let mut apply = |current: Option<StoredValue>| {
        let initial = match current {
            Some(StoredValue::Integer(value)) => value as f64,
            Some(StoredValue::Pickle(value)) => match pickle::decode(&value)? {
                Value::Number(value) => value
                    .as_i64()
                    .map(|value| value as f64)
                    .or_else(|| value.as_u64().map(|value| value as f64))
                    .unwrap_or_default(),
                _ => 0.0,
            },
            _ => 0.0,
        };
        let value = initial + amount;
        let stored = if value.fract() == 0.0 && value >= i64::MIN as f64 && value <= i64::MAX as f64
        {
            StoredValue::Integer(value as i64)
        } else {
            StoredValue::Float(value)
        };
        result = Some(value);
        Ok((stored, ttl.map(|ttl| unix_now() + ttl.as_secs_f64())))
    };
    store.update(key, unix_now(), &mut apply)?;
    result.ok_or(Error::InvalidEntry)
}

fn payload(value: StoredValue) -> Result<Option<Vec<u8>>, Error> {
    match value {
        StoredValue::Bytes(value) if value.is_empty() => Ok(None),
        StoredValue::Bytes(value) => Ok(Some(value)),
        StoredValue::Text(value) if value.is_empty() => Ok(None),
        StoredValue::Text(value) => Ok(Some(value.into_bytes())),
        StoredValue::Integer(0) => Ok(None),
        StoredValue::Integer(value) => Ok(Some(value.to_string().into_bytes())),
        StoredValue::Float(0.0) => Ok(None),
        StoredValue::Float(value) => serde_json::to_vec(&value)
            .map(Some)
            .map_err(|_| Error::InvalidEntry),
        StoredValue::Pickle(value) => {
            let value = pickle::decode(&value)?;
            if is_falsy(&value) {
                Ok(None)
            } else {
                serde_json::to_vec(&value)
                    .map(Some)
                    .map_err(|_| Error::InvalidEntry)
            }
        }
    }
}

fn is_falsy(value: &Value) -> bool {
    match value {
        Value::Null | Value::Bool(false) => true,
        Value::Number(value) => value.as_f64().is_some_and(|value| value == 0.0),
        Value::String(value) => value.is_empty(),
        Value::Array(value) => value.is_empty(),
        Value::Object(value) => value.is_empty(),
        Value::Bool(true) => false,
    }
}

fn unix_now() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs_f64()
}
