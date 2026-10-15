use std::{sync::Arc, time::Duration};

use crate::{
    BaseCache, BatchCache, BatchEntry, BulkDeleteCache, CacheContext, ClaimCache, CounterCache,
    DeleteCache, Error, FlushCache, IncrementOperation, SetCache, TtlCache,
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
    delete_batch_size: usize,
}

/// `DEFAULT_MAX_REDIS_BATCH_CACHE_SIZE`.
pub const DEFAULT_DELETE_BATCH_SIZE: usize = 1000;

impl<L1, L2> DualCache<L1, L2> {
    pub fn new(l1: Arc<L1>, l2: Arc<L2>) -> Self {
        Self {
            l1,
            l2,
            read_policy: ReadPolicy::default(),
            write_policy: WritePolicy::default(),
            remote_failure_policy: RemoteFailurePolicy::default(),
            promotion_ttl: None,
            delete_batch_size: DEFAULT_DELETE_BATCH_SIZE,
        }
    }

    pub fn with_delete_batch_size(self, delete_batch_size: usize) -> Self {
        Self {
            delete_batch_size,
            ..self
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

    fn promotion_context<C: CacheContext>(&self, context: &C) -> C {
        context.with_ttl(self.promotion_ttl.or(context.ttl()))
    }
}

impl<V, C, L1, L2> DualCache<L1, L2>
where
    V: Clone + Send + Sync + 'static,
    C: CacheContext,
    L1: BaseCache<Value = V, Context = C>,
    L2: BaseCache<Value = V, Context = C>,
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
        context: &C,
        mut entries: Vec<BatchEntry<V>>,
        missing: Vec<usize>,
        remote: Vec<BatchEntry<V>>,
    ) -> Result<Vec<BatchEntry<V>>, Error> {
        if missing.len() != remote.len() {
            return Err(Error::Unavailable);
        }
        for (index, entry) in missing.into_iter().zip(remote) {
            if let BatchEntry::Hit(value) = &entry {
                let promotion_context = self.promotion_context(context);
                self.l1
                    .set_cache(&keys[index], value.clone(), &promotion_context)?;
            }
            entries[index] = entry;
        }
        Ok(entries)
    }
}

impl<V, C, L1, L2> BaseCache for DualCache<L1, L2>
where
    V: Clone + Send + Sync + 'static,
    C: CacheContext,
    L1: BaseCache<Value = V, Context = C>,
    L2: BaseCache<Value = V, Context = C>,
{
    type Value = V;
    type Context = C;

    fn get_ttl(&self, context: &Self::Context) -> Option<Duration> {
        self.l2.get_ttl(context)
    }

    fn set_cache(&self, key: &str, value: V, context: &C) -> Result<(), Error> {
        if self.writes_remote() {
            self.remote(self.l2.set_cache(key, value.clone(), context))?;
        }
        self.l1.set_cache(key, value, context)
    }

    fn get_cache(&self, key: &str, context: &C) -> Result<Option<V>, Error> {
        if let Some(value) = self.l1.get_cache(key, context)? {
            return Ok(Some(value));
        }
        if !self.reads_remote() {
            return Ok(None);
        }
        let value = self.remote(self.l2.get_cache(key, context))?.flatten();
        if let Some(value) = &value {
            let promotion_context = self.promotion_context(context);
            self.l1.set_cache(key, value.clone(), &promotion_context)?;
        }
        Ok(value)
    }

    async fn async_set_cache(&self, key: &str, value: V, context: C) -> Result<(), Error> {
        if self.writes_remote() {
            self.remote(
                self.l2
                    .async_set_cache(key, value.clone(), context.clone())
                    .await,
            )?;
        }
        self.l1.async_set_cache(key, value, context).await
    }

    async fn async_get_cache(&self, key: &str, context: &C) -> Result<Option<V>, Error> {
        if let Some(value) = self.l1.async_get_cache(key, context).await? {
            return Ok(Some(value));
        }
        if !self.reads_remote() {
            return Ok(None);
        }
        let value = self
            .remote(self.l2.async_get_cache(key, context).await)?
            .flatten();
        if let Some(value) = &value {
            self.l1
                .async_set_cache(key, value.clone(), self.promotion_context(context))
                .await?;
        }
        Ok(value)
    }

    async fn async_set_cache_pipeline(
        &self,
        entries: Vec<(String, V)>,
        context: C,
    ) -> Result<(), Error> {
        if self.writes_remote() {
            self.remote(
                self.l2
                    .async_set_cache_pipeline(entries.clone(), context.clone())
                    .await,
            )?;
        }
        self.l1.async_set_cache_pipeline(entries, context).await
    }
}

impl<V, C, L1, L2> BatchCache for DualCache<L1, L2>
where
    V: Clone + Send + Sync + 'static,
    C: CacheContext,
    L1: BatchCache<Value = V, Context = C>,
    L2: BatchCache<Value = V, Context = C>,
{
    fn batch_get_cache(&self, keys: &[String], context: &C) -> Result<Vec<BatchEntry<V>>, Error> {
        let entries = self.l1.batch_get_cache(keys, context)?;
        let missing = Self::missing(&entries);
        if missing.is_empty() || !self.reads_remote() {
            return Ok(entries);
        }
        let remote_keys = missing
            .iter()
            .map(|index| keys[*index].clone())
            .collect::<Vec<_>>();
        match self.remote(self.l2.batch_get_cache(&remote_keys, context))? {
            Some(remote) => self.merge_batch(keys, context, entries, missing, remote),
            None => Ok(entries),
        }
    }

    async fn async_batch_get_cache(
        &self,
        keys: Vec<String>,
        context: C,
    ) -> Result<Vec<BatchEntry<V>>, Error> {
        let entries = self
            .l1
            .async_batch_get_cache(keys.clone(), context.clone())
            .await?;
        let missing = Self::missing(&entries);
        if missing.is_empty() || !self.reads_remote() {
            return Ok(entries);
        }
        let remote_keys = missing.iter().map(|index| keys[*index].clone()).collect();
        match self.remote(
            self.l2
                .async_batch_get_cache(remote_keys, context.clone())
                .await,
        )? {
            Some(remote) => self.merge_batch(&keys, &context, entries, missing, remote),
            None => Ok(entries),
        }
    }
}

impl<V, C, L1, L2> DeleteCache for DualCache<L1, L2>
where
    V: Clone + Send + Sync + 'static,
    C: CacheContext,
    L1: DeleteCache<Value = V, Context = C>,
    L2: DeleteCache<Value = V, Context = C>,
{
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
}

impl<V, C, L1, L2> FlushCache for DualCache<L1, L2>
where
    V: Clone + Send + Sync + 'static,
    C: CacheContext,
    L1: FlushCache<Value = V, Context = C>,
    L2: FlushCache<Value = V, Context = C>,
{
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
}

impl<C, L1, L2> DualCache<L1, L2>
where
    C: CacheContext,
    L1: BaseCache<Value = f64, Context = C>,
{
    /// Python's `local_only=True` increment: the local tier alone, read then written back.
    fn increment_local(&self, key: &str, amount: f64, context: &C) -> Result<f64, Error> {
        let value = self.l1.get_cache(key, context)?.unwrap_or(0.0) + amount;
        self.l1.set_cache(key, value, context)?;
        Ok(value)
    }
}

impl<C, L1, L2> CounterCache for DualCache<L1, L2>
where
    C: CacheContext,
    L1: BaseCache<Value = f64, Context = C>,
    L2: CounterCache<Value = f64, Context = C>,
{
    fn increment_cache(&self, key: &str, amount: f64, context: C) -> Result<f64, Error> {
        if !self.writes_remote() {
            return self.increment_local(key, amount, &context);
        }
        let value = self.l2.increment_cache(key, amount, context.clone())?;
        self.l1.set_cache(key, value, &context)?;
        Ok(value)
    }

    async fn async_increment(
        &self,
        key: &str,
        amount: f64,
        context: C,
        refresh_ttl: bool,
    ) -> Result<f64, Error> {
        if !self.writes_remote() {
            return self.increment_local(key, amount, &context);
        }
        let value = self
            .l2
            .async_increment(key, amount, context.clone(), refresh_ttl)
            .await?;
        self.l1.async_set_cache(key, value, context).await?;
        Ok(value)
    }

    /// `async_increment_cache_pipeline`, L2-first like single increments: the local tier takes
    /// each remote result.
    async fn async_increment_pipeline(
        &self,
        operations: Vec<IncrementOperation>,
    ) -> Result<Vec<f64>, Error>
    where
        C: Default,
    {
        if !self.writes_remote() {
            return operations
                .iter()
                .map(|operation| {
                    let context = C::default().with_ttl(operation.ttl);
                    self.increment_local(&operation.key, operation.amount, &context)
                })
                .collect();
        }
        let values = self.l2.async_increment_pipeline(operations.clone()).await?;
        if values.len() != operations.len() {
            return Err(Error::Unavailable);
        }
        for (operation, value) in operations.iter().zip(&values) {
            self.l1
                .async_set_cache(&operation.key, *value, C::default().with_ttl(operation.ttl))
                .await?;
        }
        Ok(values)
    }
}

/// `async_set_cache_sadd`: local set first, then the remote one unless writes stay local.
impl<V, C, S, L1, L2> SetCache for DualCache<L1, L2>
where
    V: Clone + Send + Sync + 'static,
    C: CacheContext,
    S: Clone + Send + Sync + 'static,
    L1: SetCache<Value = V, Context = C, SetValue = S>,
    L2: SetCache<Value = V, Context = C, SetValue = S>,
{
    type SetValue = S;
    type SetResult = ();

    async fn async_set_cache_sadd(
        &self,
        key: &str,
        values: Vec<S>,
        ttl: Option<Duration>,
    ) -> Result<(), Error> {
        self.l1
            .async_set_cache_sadd(key, values.clone(), ttl)
            .await?;
        if self.writes_remote() {
            self.l2.async_set_cache_sadd(key, values, ttl).await?;
        }
        Ok(())
    }
}

/// `async_delete_cache_keys`: every key leaves the local tier, then the remote tier in chunks
/// of `DEFAULT_MAX_REDIS_BATCH_CACHE_SIZE`, since Redis takes a chunk as one command.
impl<V, C, L1, L2> BulkDeleteCache for DualCache<L1, L2>
where
    V: Clone + Send + Sync + 'static,
    C: CacheContext,
    L1: DeleteCache<Value = V, Context = C>,
    L2: BulkDeleteCache<Value = V, Context = C>,
{
    async fn delete_cache_keys(&self, keys: Vec<String>) -> Result<usize, Error> {
        for key in &keys {
            self.l1.delete_cache(key)?;
        }
        let mut deleted = 0;
        for chunk in keys.chunks(self.delete_batch_size.max(1)) {
            deleted += self.l2.delete_cache_keys(chunk.to_vec()).await?;
        }
        Ok(deleted)
    }
}

/// `async_get_ttl`: the local TTL, or the remote one when the local tier has none.
impl<V, C, L1, L2> TtlCache for DualCache<L1, L2>
where
    V: Clone + Send + Sync + 'static,
    C: CacheContext,
    L1: TtlCache<Value = V, Context = C>,
    L2: TtlCache<Value = V, Context = C>,
{
    async fn async_get_ttl(&self, key: &str) -> Result<Option<Duration>, Error> {
        match self.l1.async_get_ttl(key).await? {
            Some(ttl) => Ok(Some(ttl)),
            None => self.l2.async_get_ttl(key).await,
        }
    }
}

impl<V, C, L1, L2> ClaimCache for DualCache<L1, L2>
where
    V: Clone + PartialEq + Send + Sync + 'static,
    C: CacheContext,
    L1: ClaimCache<Value = V, Context = C>,
    L2: ClaimCache<Value = V, Context = C>,
{
    fn claim_cache(&self, key: &str, candidate: V, eligible: &[V], context: C) -> Result<V, Error> {
        match self.remote(
            self.l2
                .claim_cache(key, candidate.clone(), eligible, context.clone()),
        )? {
            Some(winner) => {
                self.l1.set_cache(key, winner.clone(), &context)?;
                Ok(winner)
            }
            None => self.l1.claim_cache(key, candidate, eligible, context),
        }
    }

    async fn async_claim_cache(
        &self,
        key: &str,
        candidate: V,
        eligible: Vec<V>,
        context: C,
    ) -> Result<V, Error> {
        match self.remote(
            self.l2
                .async_claim_cache(key, candidate.clone(), eligible.clone(), context.clone())
                .await,
        )? {
            Some(winner) => {
                self.l1
                    .async_set_cache(key, winner.clone(), context)
                    .await?;
                Ok(winner)
            }
            None => {
                self.l1
                    .async_claim_cache(key, candidate, eligible, context)
                    .await
            }
        }
    }
}
