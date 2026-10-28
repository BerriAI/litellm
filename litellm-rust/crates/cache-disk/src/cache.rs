use std::{
    path::Path,
    sync::Arc,
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use litellm_cache::{
    BaseCache, BatchCache, BatchEntry, CacheCodec, CounterCache, DeleteCache, DisconnectCache,
    Error, ExactCacheContext, FlushCache,
};

use crate::{DiskStore, DiskcacheSqliteStore, PythonDiskCacheAdapter, StoredValue, ValueAdapter};

pub struct DiskCache<S, D = DiskcacheSqliteStore, A = PythonDiskCacheAdapter> {
    store: Arc<D>,
    adapter: Arc<A>,
    codec: S,
}

impl<S: CacheCodec> DiskCache<S> {
    pub fn open(directory: impl AsRef<Path>, codec: S) -> Result<Self, Error> {
        Ok(Self {
            store: Arc::new(DiskcacheSqliteStore::open(directory)?),
            adapter: Arc::new(PythonDiskCacheAdapter),
            codec,
        })
    }
}

impl<S: CacheCodec, D: DiskStore> DiskCache<S, D, PythonDiskCacheAdapter> {
    pub fn with_store(store: D, codec: S) -> Self {
        Self {
            store: Arc::new(store),
            adapter: Arc::new(PythonDiskCacheAdapter),
            codec,
        }
    }
}

impl<S: CacheCodec, D: DiskStore, A: ValueAdapter> DiskCache<S, D, A> {
    pub fn with_adapter(store: D, adapter: A, codec: S) -> Self {
        Self {
            store: Arc::new(store),
            adapter: Arc::new(adapter),
            codec,
        }
    }

    pub fn directory(&self) -> &Path {
        self.store.directory()
    }

    fn decode_stored(&self, value: StoredValue) -> Result<Option<S::Value>, Error> {
        let Some(bytes) = self.adapter.read(value)? else {
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

impl<S: CacheCodec, D: DiskStore, A: ValueAdapter> BaseCache for DiskCache<S, D, A> {
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
        let value = self.adapter.write(self.codec.encode(&value)?);
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
        let value = self.adapter.write(self.codec.encode(&value)?);
        let ttl = context.ttl;
        let key = key.to_string();
        Self::run_blocking(Arc::clone(&self.store), move |store| {
            let expire_time = ttl.map(|ttl| unix_now() + ttl.as_secs_f64());
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
                    .map(|value| (key, self.adapter.write(value)))
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
}

impl<S: CacheCodec, D: DiskStore, A: ValueAdapter> BatchCache for DiskCache<S, D, A> {
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

impl<S: CacheCodec, D: DiskStore, A: ValueAdapter> DeleteCache for DiskCache<S, D, A> {
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

impl<S: CacheCodec, D: DiskStore, A: ValueAdapter> FlushCache for DiskCache<S, D, A> {
    fn flush_cache(&self) -> Result<(), Error> {
        self.store.clear()
    }

    async fn async_flush_cache(&self) -> Result<(), Error> {
        Self::run_blocking(Arc::clone(&self.store), |store| store.clear()).await
    }
}

impl<S: CacheCodec, D: DiskStore, A: ValueAdapter> DisconnectCache for DiskCache<S, D, A> {
    async fn disconnect(&self) -> Result<(), Error> {
        Ok(())
    }
}

impl<S: CacheCodec, D: DiskStore, A: ValueAdapter> CounterCache for DiskCache<S, D, A> {
    fn increment_cache(
        &self,
        key: &str,
        amount: f64,
        context: ExactCacheContext,
    ) -> Result<f64, Error> {
        increment(
            self.adapter.as_ref(),
            self.store.as_ref(),
            key,
            amount,
            context.ttl,
        )
    }

    async fn async_increment(
        &self,
        key: &str,
        amount: f64,
        context: ExactCacheContext,
        _refresh_ttl: bool,
    ) -> Result<f64, Error> {
        let key = key.to_string();
        let adapter = Arc::clone(&self.adapter);
        Self::run_blocking(Arc::clone(&self.store), move |store| {
            increment(adapter.as_ref(), store, &key, amount, context.ttl)
        })
        .await
    }
}

fn increment<A: ValueAdapter, D: DiskStore>(
    adapter: &A,
    store: &D,
    key: &str,
    amount: f64,
    ttl: Option<Duration>,
) -> Result<f64, Error> {
    let mut result = None;
    let mut apply = |current: Option<StoredValue>| {
        let initial = adapter.counter_seed(current)?;
        let value = initial + amount;
        let stored = adapter.counter_value(value);
        result = Some(value);
        Ok((stored, ttl.map(|ttl| unix_now() + ttl.as_secs_f64())))
    };
    store.update(key, unix_now(), &mut apply)?;
    result.ok_or(Error::InvalidEntry)
}

fn unix_now() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs_f64()
}
