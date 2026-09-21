use std::sync::Arc;

use crate::{BaseCache, CacheConnectionResult, CacheKwargs, ClaimCache, CounterCache, Error};

pub struct DualCache<L1, L2> {
    l1: Arc<L1>,
    l2: Arc<L2>,
}

impl<L1, L2> DualCache<L1, L2> {
    pub fn new(l1: Arc<L1>, l2: Arc<L2>) -> Self {
        Self { l1, l2 }
    }
}

impl<V, L1, L2> BaseCache for DualCache<L1, L2>
where
    V: Clone + Send + Sync + 'static,
    L1: BaseCache<Value = V>,
    L2: BaseCache<Value = V>,
{
    type Value = V;

    fn default_ttl(&self) -> std::time::Duration {
        self.l2.default_ttl()
    }

    fn set_cache(&self, key: &str, value: V, kwargs: CacheKwargs) -> Result<(), Error> {
        self.l2.set_cache(key, value.clone(), kwargs.clone())?;
        self.l1.set_cache(key, value, kwargs)
    }

    fn get_cache(&self, key: &str, kwargs: &CacheKwargs) -> Result<Option<V>, Error> {
        if let Some(value) = self.l1.get_cache(key, kwargs)? {
            return Ok(Some(value));
        }
        let value = self.l2.get_cache(key, kwargs)?;
        if let Some(value) = &value {
            self.l1.set_cache(key, value.clone(), kwargs.clone())?;
        }
        Ok(value)
    }

    fn delete_cache(&self, key: &str) -> Result<(), Error> {
        self.l2.delete_cache(key)?;
        self.l1.delete_cache(key)
    }

    fn flush_cache(&self) -> Result<(), Error> {
        self.l2.flush_cache()?;
        self.l1.flush_cache()
    }

    async fn async_flush_cache(&self) -> Result<(), Error> {
        self.l2.async_flush_cache().await?;
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
        match self
            .l2
            .claim_cache(key, candidate.clone(), eligible, kwargs.clone())
        {
            Ok(winner) => {
                self.l1.set_cache(key, winner.clone(), kwargs)?;
                Ok(winner)
            }
            Err(_) => self.l1.claim_cache(key, candidate, eligible, kwargs),
        }
    }
}
