use std::future::Future;

use crate::{BaseCache, CacheKwargs, Error};

#[derive(Clone, Debug, PartialEq)]
pub struct IncrementOperation {
    pub key: String,
    pub amount: f64,
    pub ttl: Option<std::time::Duration>,
}

pub trait CounterCache: BaseCache<Value = f64> {
    fn increment_cache(&self, key: &str, amount: f64, kwargs: CacheKwargs) -> Result<f64, Error>;

    fn async_increment_cache(
        &self,
        key: &str,
        amount: f64,
        kwargs: CacheKwargs,
    ) -> impl Future<Output = Result<f64, Error>> + Send {
        async move { self.increment_cache(key, amount, kwargs) }
    }
}

pub trait ClaimCache: BaseCache
where
    Self::Value: PartialEq,
{
    fn claim_cache(
        &self,
        key: &str,
        candidate: Self::Value,
        eligible: &[Self::Value],
        kwargs: CacheKwargs,
    ) -> Result<Self::Value, Error>;

    fn async_claim_cache(
        &self,
        key: &str,
        candidate: Self::Value,
        eligible: Vec<Self::Value>,
        kwargs: CacheKwargs,
    ) -> impl Future<Output = Result<Self::Value, Error>> + Send {
        async move { self.claim_cache(key, candidate, &eligible, kwargs) }
    }
}
