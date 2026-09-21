use std::{sync::Arc, time::Duration};

use crate::{
    BaseCache, BatchEntry, CacheConnectionResult, CacheKwargs, ClaimCache, CounterCache, Error,
};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ReadPolicy {
    #[default]
    LocalThenRemote,
    LocalOnly,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum WritePolicy {
    #[default]
    Both,
    LocalOnly,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum RemoteFailurePolicy {
    #[default]
    Propagate,
    UseLocal,
}

pub struct DualCache<L1, L2> {
    l1: Arc<L1>,
    l2: Arc<L2>,
    read_policy: ReadPolicy,
    write_policy: WritePolicy,
    remote_failure_policy: RemoteFailurePolicy,
    promotion_ttl: Option<Duration>,
}

impl<L1, L2> DualCache<L1, L2> {
    pub fn new(l1: Arc<L1>, l2: Arc<L2>) -> Self {
        Self {
            l1,
            l2,
            read_policy: ReadPolicy::default(),
            write_policy: WritePolicy::default(),
            remote_failure_policy: RemoteFailurePolicy::default(),
            promotion_ttl: None,
        }
    }

    pub fn with_read_policy(self, read_policy: ReadPolicy) -> Self {
        Self {
            read_policy,
            ..self
        }
    }

    pub fn with_write_policy(self, write_policy: WritePolicy) -> Self {
        Self {
            write_policy,
            ..self
        }
    }

    pub fn with_remote_failure_policy(self, remote_failure_policy: RemoteFailurePolicy) -> Self {
        Self {
            remote_failure_policy,
            ..self
        }
    }

    pub fn with_promotion_ttl(self, promotion_ttl: Duration) -> Self {
        Self {
            promotion_ttl: Some(promotion_ttl),
            ..self
        }
    }

    fn reads_remote(&self) -> bool {
        self.read_policy == ReadPolicy::LocalThenRemote
    }

    fn writes_remote(&self) -> bool {
        self.write_policy == WritePolicy::Both
    }

    fn remote<T>(&self, result: Result<T, Error>) -> Result<Option<T>, Error> {
        match result {
            Ok(value) => Ok(Some(value)),
            Err(Error::Unavailable)
                if self.remote_failure_policy == RemoteFailurePolicy::UseLocal =>
            {
                Ok(None)
            }
            Err(error) => Err(error),
        }
    }

    fn promotion_kwargs(&self, kwargs: &CacheKwargs) -> CacheKwargs {
        CacheKwargs {
            ttl: self.promotion_ttl.or(kwargs.ttl),
            extras: kwargs.extras.clone(),
        }
    }
}

impl<V, L1, L2> DualCache<L1, L2>
where
    V: Clone + Send + Sync + 'static,
    L1: BaseCache<Value = V>,
    L2: BaseCache<Value = V>,
{
    fn missing(entries: &[BatchEntry<V>]) -> Vec<usize> {
        entries
            .iter()
            .enumerate()
            .filter_map(|(index, entry)| (!matches!(entry, BatchEntry::Hit(_))).then_some(index))
            .collect()
    }

    fn merge_batch(
        &self,
        keys: &[String],
        kwargs: &CacheKwargs,
        mut entries: Vec<BatchEntry<V>>,
        missing: Vec<usize>,
        remote: Vec<BatchEntry<V>>,
    ) -> Result<Vec<BatchEntry<V>>, Error> {
        if missing.len() != remote.len() {
            return Err(Error::Unavailable);
        }
        for (index, entry) in missing.into_iter().zip(remote) {
            if let BatchEntry::Hit(value) = &entry {
                self.l1
                    .set_cache(&keys[index], value.clone(), self.promotion_kwargs(kwargs))?;
            }
            entries[index] = entry;
        }
        Ok(entries)
    }
}

impl<V, L1, L2> BaseCache for DualCache<L1, L2>
where
    V: Clone + Send + Sync + 'static,
    L1: BaseCache<Value = V>,
    L2: BaseCache<Value = V>,
{
    type Value = V;

    fn default_ttl(&self) -> Duration {
        self.l2.default_ttl()
    }

    fn set_cache(&self, key: &str, value: V, kwargs: CacheKwargs) -> Result<(), Error> {
        if self.writes_remote() {
            self.remote(self.l2.set_cache(key, value.clone(), kwargs.clone()))?;
        }
        self.l1.set_cache(key, value, kwargs)
    }

    fn get_cache(&self, key: &str, kwargs: &CacheKwargs) -> Result<Option<V>, Error> {
        if let Some(value) = self.l1.get_cache(key, kwargs)? {
            return Ok(Some(value));
        }
        if !self.reads_remote() {
            return Ok(None);
        }
        let value = self.remote(self.l2.get_cache(key, kwargs))?.flatten();
        if let Some(value) = &value {
            self.l1
                .set_cache(key, value.clone(), self.promotion_kwargs(kwargs))?;
        }
        Ok(value)
    }

    fn get_cache_batch(
        &self,
        keys: &[String],
        kwargs: &CacheKwargs,
    ) -> Result<Vec<BatchEntry<V>>, Error> {
        let entries = self.l1.get_cache_batch(keys, kwargs)?;
        let missing = Self::missing(&entries);
        if missing.is_empty() || !self.reads_remote() {
            return Ok(entries);
        }
        let remote_keys = missing
            .iter()
            .map(|index| keys[*index].clone())
            .collect::<Vec<_>>();
        match self.remote(self.l2.get_cache_batch(&remote_keys, kwargs))? {
            Some(remote) => self.merge_batch(keys, kwargs, entries, missing, remote),
            None => Ok(entries),
        }
    }

    async fn async_set_cache(&self, key: &str, value: V, kwargs: CacheKwargs) -> Result<(), Error> {
        if self.writes_remote() {
            self.remote(
                self.l2
                    .async_set_cache(key, value.clone(), kwargs.clone())
                    .await,
            )?;
        }
        self.l1.async_set_cache(key, value, kwargs).await
    }

    async fn async_get_cache(&self, key: &str, kwargs: &CacheKwargs) -> Result<Option<V>, Error> {
        if let Some(value) = self.l1.async_get_cache(key, kwargs).await? {
            return Ok(Some(value));
        }
        if !self.reads_remote() {
            return Ok(None);
        }
        let value = self
            .remote(self.l2.async_get_cache(key, kwargs).await)?
            .flatten();
        if let Some(value) = &value {
            self.l1
                .async_set_cache(key, value.clone(), self.promotion_kwargs(kwargs))
                .await?;
        }
        Ok(value)
    }

    async fn async_get_cache_batch(
        &self,
        keys: Vec<String>,
        kwargs: CacheKwargs,
    ) -> Result<Vec<BatchEntry<V>>, Error> {
        let entries = self
            .l1
            .async_get_cache_batch(keys.clone(), kwargs.clone())
            .await?;
        let missing = Self::missing(&entries);
        if missing.is_empty() || !self.reads_remote() {
            return Ok(entries);
        }
        let remote_keys = missing.iter().map(|index| keys[*index].clone()).collect();
        match self.remote(
            self.l2
                .async_get_cache_batch(remote_keys, kwargs.clone())
                .await,
        )? {
            Some(remote) => self.merge_batch(&keys, &kwargs, entries, missing, remote),
            None => Ok(entries),
        }
    }

    async fn async_set_cache_pipeline(
        &self,
        cache_list: Vec<(String, V)>,
        kwargs: CacheKwargs,
    ) -> Result<(), Error> {
        if self.writes_remote() {
            self.remote(
                self.l2
                    .async_set_cache_pipeline(cache_list.clone(), kwargs.clone())
                    .await,
            )?;
        }
        self.l1.async_set_cache_pipeline(cache_list, kwargs).await
    }

    fn delete_cache(&self, key: &str) -> Result<(), Error> {
        if self.writes_remote() {
            self.remote(self.l2.delete_cache(key))?;
        }
        self.l1.delete_cache(key)
    }

    async fn async_delete_cache(&self, key: &str) -> Result<(), Error> {
        if self.writes_remote() {
            self.remote(self.l2.async_delete_cache(key).await)?;
        }
        self.l1.async_delete_cache(key).await
    }

    fn flush_cache(&self) -> Result<(), Error> {
        if self.writes_remote() {
            self.remote(self.l2.flush_cache())?;
        }
        self.l1.flush_cache()
    }

    async fn async_flush_cache(&self) -> Result<(), Error> {
        if self.writes_remote() {
            self.remote(self.l2.async_flush_cache().await)?;
        }
        self.l1.async_flush_cache().await
    }

    async fn disconnect(&self) -> Result<(), Error> {
        self.l2.disconnect().await?;
        self.l1.disconnect().await
    }

    async fn test_connection(&self) -> Result<CacheConnectionResult, Error> {
        self.l2.test_connection().await
    }
}

impl<L1, L2> CounterCache for DualCache<L1, L2>
where
    L1: BaseCache<Value = f64>,
    L2: CounterCache,
{
    fn increment_cache(&self, key: &str, amount: f64, kwargs: CacheKwargs) -> Result<f64, Error> {
        let value = self.l2.increment_cache(key, amount, kwargs.clone())?;
        self.l1.set_cache(key, value, kwargs)?;
        Ok(value)
    }

    async fn async_increment_cache(
        &self,
        key: &str,
        amount: f64,
        kwargs: CacheKwargs,
    ) -> Result<f64, Error> {
        let value = self
            .l2
            .async_increment_cache(key, amount, kwargs.clone())
            .await?;
        self.l1.async_set_cache(key, value, kwargs).await?;
        Ok(value)
    }
}

impl<V, L1, L2> ClaimCache for DualCache<L1, L2>
where
    V: Clone + PartialEq + Send + Sync + 'static,
    L1: ClaimCache<Value = V>,
    L2: ClaimCache<Value = V>,
{
    fn claim_cache(
        &self,
        key: &str,
        candidate: V,
        eligible: &[V],
        kwargs: CacheKwargs,
    ) -> Result<V, Error> {
        match self.remote(
            self.l2
                .claim_cache(key, candidate.clone(), eligible, kwargs.clone()),
        )? {
            Some(winner) => {
                self.l1.set_cache(key, winner.clone(), kwargs)?;
                Ok(winner)
            }
            None => self.l1.claim_cache(key, candidate, eligible, kwargs),
        }
    }

    async fn async_claim_cache(
        &self,
        key: &str,
        candidate: V,
        eligible: Vec<V>,
        kwargs: CacheKwargs,
    ) -> Result<V, Error> {
        match self.remote(
            self.l2
                .async_claim_cache(key, candidate.clone(), eligible.clone(), kwargs.clone())
                .await,
        )? {
            Some(winner) => {
                self.l1.async_set_cache(key, winner.clone(), kwargs).await?;
                Ok(winner)
            }
            None => {
                self.l1
                    .async_claim_cache(key, candidate, eligible, kwargs)
                    .await
            }
        }
    }
}
