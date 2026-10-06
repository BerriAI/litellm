use std::time::Duration;

use litellm_cache::{
    BaseCache, Error,
    semantic::{SemanticCache, SemanticLookup},
};
use serde_json::Value;

use crate::{CacheEntry, CacheKey, ResponseCache};

impl<B> ResponseCache<B>
where
    B: BaseCache<Value = CacheEntry> + SemanticCache,
    B::Context: Default + PartialEq,
{
    /// `lookup` plus the similarity the semantic backend reports. Freshness applies to the
    /// value only: Python stamps the similarity before its max-age check.
    pub fn lookup_semantic(
        &self,
        key: &CacheKey,
        context: &B::Context,
        max_age: Option<Duration>,
        now: Duration,
    ) -> Result<SemanticLookup<Value>, Error> {
        let lookup = self
            .backend
            .get_cache_with_similarity(key.as_str(), context);
        fresh_semantic(lookup, now, max_age)
    }

    pub async fn async_lookup_semantic(
        &self,
        key: &CacheKey,
        context: &B::Context,
        max_age: Option<Duration>,
        now: Duration,
    ) -> Result<SemanticLookup<Value>, Error> {
        let lookup = self
            .backend
            .async_get_cache_with_similarity(key.as_str(), context)
            .await;
        fresh_semantic(lookup, now, max_age)
    }
}

fn fresh_semantic(
    lookup: Result<SemanticLookup<CacheEntry>, Error>,
    now: Duration,
    max_age: Option<Duration>,
) -> Result<SemanticLookup<Value>, Error> {
    match lookup {
        Ok(lookup) => Ok(SemanticLookup {
            value: lookup
                .value
                .and_then(|entry| entry.into_fresh(now, max_age)),
            similarity: lookup.similarity,
        }),
        Err(Error::InvalidEntry) => Ok(SemanticLookup::miss(None)),
        Err(error) => Err(error),
    }
}
