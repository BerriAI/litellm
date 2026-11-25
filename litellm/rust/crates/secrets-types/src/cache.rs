use std::{future::Future, hash::Hash, sync::Arc, time::Duration};

use moka::future::Cache;
use tokio::sync::Mutex;

#[derive(Clone)]
pub struct SecretCache<K, V> {
    entries: Cache<K, Arc<Mutex<Option<V>>>>,
}

impl<K, V> SecretCache<K, V>
where
    K: Eq + Hash + Clone + Send + Sync + 'static,
    V: Clone + Send + Sync + 'static,
{
    pub fn new(capacity: u64, ttl: Duration) -> Self {
        Self {
            entries: Cache::builder()
                .max_capacity(capacity)
                .time_to_live(ttl)
                .support_invalidation_closures()
                .build(),
        }
    }

    pub async fn read<E>(
        &self,
        key: K,
        load: impl Future<Output = Result<Option<V>, E>>,
    ) -> Result<Option<V>, E> {
        let entry = self
            .entries
            .get_with(key, async { Arc::new(Mutex::new(None)) })
            .await;
        let mut value = entry.lock().await;
        if value.is_some() {
            return Ok(value.clone());
        }
        // Invalidated loads only populate their detached entry, never the cache's replacement.
        let loaded = load.await?;
        *value = loaded.clone();
        Ok(loaded)
    }

    pub async fn invalidate(&self, key: &K) {
        self.entries.invalidate(key).await;
    }

    pub async fn refresh<E>(
        &self,
        key: K,
        load: impl Future<Output = Result<Option<V>, E>>,
    ) -> Result<Option<V>, E> {
        let entry = Arc::new(Mutex::new(None));
        let mut value = entry.lock().await;
        self.entries.insert(key, entry.clone()).await;
        let loaded = load.await?;
        *value = loaded.clone();
        Ok(loaded)
    }

    pub async fn insert(&self, key: K, value: V) {
        self.entries
            .insert(key, Arc::new(Mutex::new(Some(value))))
            .await;
    }

    pub fn invalidate_where(&self, predicate: impl Fn(&K) -> bool + Send + Sync + 'static) {
        self.entries
            .invalidate_entries_if(move |key, _| predicate(key))
            .expect("invalidation closures are enabled");
    }
}
