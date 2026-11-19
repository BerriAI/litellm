use std::{collections::HashSet, hash::Hash, time::Duration};

use litellm_cache::{
    BaseCache, BatchCache, ClaimCache, CounterCache, DeleteCache, DisconnectCache, Error,
    ExactCacheContext, FlushCache, SetCache, TtlCache,
};

use crate::cache::{Expiry, InMemoryCache};

impl<V: Clone + Send + Sync + 'static> BaseCache for InMemoryCache<V> {
    type Value = V;
    type Context = ExactCacheContext;

    fn get_ttl(&self, context: &Self::Context) -> Option<Duration> {
        context.ttl.or(Some(self.default_ttl()))
    }

    fn set_cache(
        &self,
        key: &str,
        value: Self::Value,
        context: &ExactCacheContext,
    ) -> Result<(), Error> {
        InMemoryCache::set_cache(self, key, value, Expiry::KeepExisting(context.ttl)).map(drop)
    }

    fn get_cache(&self, key: &str, _: &ExactCacheContext) -> Result<Option<Self::Value>, Error> {
        InMemoryCache::get_cache(self, key)
    }
}

impl<V> ClaimCache for InMemoryCache<V>
where
    V: Clone + PartialEq + Send + Sync + 'static,
{
    /// A held value outside `eligible` gives way to `candidate`. With no `eligible` list the held
    /// value always wins, and only a claim that agrees with it refreshes the TTL.
    fn claim_cache(
        &self,
        key: &str,
        candidate: V,
        eligible: &[V],
        context: ExactCacheContext,
    ) -> Result<V, Error> {
        self.update(key, Expiry::Refresh(context.ttl), |held| {
            let held = held.filter(|held| eligible.is_empty() || eligible.contains(held));
            Ok(match held {
                Some(held) if eligible.is_empty() && *held != candidate => (None, held.clone()),
                Some(held) => (Some(held.clone()), held.clone()),
                None => (Some(candidate.clone()), candidate),
            })
        })
    }
}

impl CounterCache for InMemoryCache<f64> {
    fn increment_cache(
        &self,
        key: &str,
        amount: f64,
        context: ExactCacheContext,
    ) -> Result<f64, Error> {
        self.update(key, Expiry::KeepExisting(context.ttl), |held| {
            let total = held.copied().unwrap_or_default() + amount;
            Ok((Some(total), total))
        })
    }
}

impl<V: Clone + Send + Sync + 'static> DisconnectCache for InMemoryCache<V> {
    async fn disconnect(&self) -> Result<(), Error> {
        Ok(())
    }
}

impl<V: Clone + Send + Sync + 'static> BatchCache for InMemoryCache<V> {}

impl<V: Clone + Send + Sync + 'static> DeleteCache for InMemoryCache<V> {
    fn delete_cache(&self, key: &str) -> Result<(), Error> {
        InMemoryCache::delete_cache(self, key)
    }
}

impl<V: Clone + Send + Sync + 'static> FlushCache for InMemoryCache<V> {
    fn flush_cache(&self) -> Result<(), Error> {
        InMemoryCache::flush_cache(self)
    }
}

impl<V: Clone + Send + Sync + 'static> TtlCache for InMemoryCache<V> {
    async fn async_get_ttl(&self, key: &str) -> Result<Option<Duration>, Error> {
        InMemoryCache::async_get_ttl(self, key).await
    }
}

impl<T> SetCache for InMemoryCache<HashSet<T>>
where
    T: Clone + Eq + Hash + Send + Sync + 'static,
{
    type SetValue = T;
    type SetResult = Vec<T>;

    async fn async_set_cache_sadd(
        &self,
        key: &str,
        values: Vec<Self::SetValue>,
        ttl: Option<Duration>,
    ) -> Result<Self::SetResult, Error> {
        self.update(key, Expiry::KeepExisting(ttl), |held| {
            let members = held.into_iter().flatten().chain(&values).cloned().collect();
            Ok((Some(members), ()))
        })?;
        Ok(values)
    }
}

#[cfg(test)]
mod tests {
    use std::{collections::HashSet, sync::Arc, time::Duration};

    use litellm_cache::{
        BaseCache, BatchCache, BatchEntry, CacheBackend, ClaimCache, CounterCache, DeleteCache,
        DisconnectCache, ExactCacheContext, FlushCache, IncrementOperation, SetCache, TtlCache,
        get_cache, set_cache,
    };
    use rstest::rstest;

    use crate::{
        Expiry, InMemoryCache,
        test_support::{Clock, at, cache, clock, secs},
    };

    const DEFAULT: Expiry = Expiry::KeepExisting(None);

    fn keep(seconds: u64) -> Expiry {
        Expiry::KeepExisting(secs(seconds))
    }

    fn ttl(seconds: u64) -> ExactCacheContext {
        ExactCacheContext { ttl: secs(seconds) }
    }

    #[rstest]
    #[tokio::test]
    async fn generic_consumers_share_typed_values_and_honor_expiration(clock: Clock) {
        let cache: CacheBackend<InMemoryCache<String>> = Arc::new(cache(&clock, 4));
        let reader = Arc::clone(&cache);
        let context = ttl(5);
        set_cache(cache.as_ref(), "sync", "first".into(), &context).unwrap();
        cache
            .batch_cache_write("async", "second".into(), context.clone())
            .await
            .unwrap();
        cache
            .async_set_cache_pipeline(vec![("batch".into(), "third".into())], context.clone())
            .await
            .unwrap();
        drop(cache);

        let written = (
            get_cache(reader.as_ref(), "sync", &context),
            reader.async_get_cache("async", &context).await,
            reader.async_get_cache("batch", &context).await,
        );
        reader.async_delete_cache("async").await.unwrap();
        let deleted = reader.async_get_cache("async", &context).await;
        at(&clock, 105);
        let expired = (
            get_cache(reader.as_ref(), "sync", &context),
            reader.async_get_cache("batch", &context).await,
        );

        assert_eq!(
            written,
            (
                Ok(Some("first".into())),
                Ok(Some("second".into())),
                Ok(Some("third".into()))
            )
        );
        assert_eq!(deleted, Ok(None));
        assert_eq!(expired, (Ok(None), Ok(None)));
    }

    #[rstest]
    #[case::context_ttl(ttl(5), secs(105))]
    #[case::default_ttl(ExactCacheContext::default(), secs(160))]
    #[tokio::test]
    async fn pipeline_writes_use_the_context_ttl_or_the_default(
        clock: Clock,
        #[case] context: ExactCacheContext,
        #[case] expected: Option<Duration>,
    ) {
        let cache = cache::<String>(&clock, 4);

        cache
            .async_set_cache_pipeline(
                vec![("a".into(), "1".into()), ("b".into(), "2".into())],
                context,
            )
            .await
            .unwrap();

        assert_eq!(cache.expires_at("a"), Ok(expected));
        assert_eq!(cache.expires_at("b"), Ok(expected));
    }

    #[rstest]
    #[tokio::test]
    async fn batch_reads_return_one_entry_per_key_and_miss_expired_ones(clock: Clock) {
        let cache = cache::<String>(&clock, 4);
        cache.set_cache("short", "a".into(), keep(1)).unwrap();
        cache.set_cache("long", "b".into(), keep(50)).unwrap();
        let keys = vec!["short".to_owned(), "missing".into(), "long".into()];

        let live = cache.batch_get_cache(&keys, &ExactCacheContext::default());
        at(&clock, 101);
        let later = cache
            .async_batch_get_cache(keys, ExactCacheContext::default())
            .await;

        assert_eq!(
            live,
            Ok(vec![
                BatchEntry::Hit("a".to_owned()),
                BatchEntry::Miss,
                BatchEntry::Hit("b".into()),
            ])
        );
        assert_eq!(
            later,
            Ok(vec![
                BatchEntry::Miss,
                BatchEntry::Miss,
                BatchEntry::Hit("b".into()),
            ])
        );
    }

    #[rstest]
    #[tokio::test]
    async fn disconnect_keeps_entries_and_an_async_flush_drops_them() {
        let cache = InMemoryCache::<String>::default();
        cache.set_cache("key", "value".into(), DEFAULT).unwrap();

        cache.disconnect().await.unwrap();
        let after_disconnect = cache.get_cache("key");
        FlushCache::async_flush_cache(&cache).await.unwrap();

        assert_eq!(after_disconnect, Ok(Some("value".into())));
        assert_eq!(cache.is_empty(), Ok(true));
    }

    #[rstest]
    #[tokio::test]
    async fn the_ttl_capability_reads_the_deadline(clock: Clock) {
        let cache = cache::<String>(&clock, 3);
        cache.set_cache("key", "1".into(), keep(20)).unwrap();

        assert_eq!(TtlCache::async_get_ttl(&cache, "key").await, Ok(secs(120)));
    }

    #[rstest]
    fn a_claim_without_an_eligible_list_keeps_the_held_value_and_its_ttl(clock: Clock) {
        let cache = cache::<String>(&clock, 4);
        let first = cache.claim_cache("affinity", "first".to_owned(), &[], ttl(10));
        at(&clock, 103);

        let second = cache.claim_cache("affinity", "second".to_owned(), &[], ttl(10));

        assert_eq!((first, second), (Ok("first".into()), Ok("first".into())));
        assert_eq!(cache.expires_at("affinity"), Ok(secs(110)));
    }

    #[rstest]
    #[case::the_held_value_is_eligible(&["first", "second"], "first")]
    #[case::the_held_value_is_not_eligible(&["second"], "second")]
    fn a_claim_with_an_eligible_list_refreshes_the_ttl_of_the_winner(
        clock: Clock,
        #[case] eligible: &[&str],
        #[case] winner: &str,
    ) {
        let cache = cache::<String>(&clock, 4);
        let eligible = eligible
            .iter()
            .copied()
            .map(String::from)
            .collect::<Vec<_>>();
        cache
            .set_cache("affinity", "first".into(), keep(10))
            .unwrap();
        at(&clock, 105);

        let claimed = cache.claim_cache("affinity", "second".to_owned(), &eligible, ttl(10));

        assert_eq!(claimed, Ok(winner.to_owned()));
        assert_eq!(cache.get_cache("affinity"), Ok(Some(winner.to_owned())));
        assert_eq!(cache.expires_at("affinity"), Ok(secs(115)));
    }

    #[rstest]
    fn claiming_a_held_key_at_capacity_keeps_other_entries(clock: Clock) {
        let cache = cache::<String>(&clock, 2);
        cache.set_cache("hot", "1".into(), keep(10)).unwrap();
        cache.set_cache("cold", "2".into(), keep(20)).unwrap();

        cache
            .claim_cache("cold", "2".into(), &[], ExactCacheContext::default())
            .unwrap();

        assert_eq!(cache.get_cache("hot"), Ok(Some("1".into())));
        assert_eq!(cache.expires_at("cold"), Ok(secs(160)));
    }

    #[rstest]
    fn counters_accumulate() {
        let cache = InMemoryCache::<f64>::default();

        let totals = (
            cache.increment_cache("counter", 1.5, ExactCacheContext::default()),
            cache.increment_cache("counter", 2.0, ExactCacheContext::default()),
        );

        assert_eq!(totals, (Ok(1.5), Ok(3.5)));
    }

    #[rstest]
    fn concurrent_increments_lose_no_update() {
        let cache = InMemoryCache::<f64>::default();

        std::thread::scope(|scope| {
            for _ in 0..8 {
                scope.spawn(|| {
                    for _ in 0..10_000 {
                        cache
                            .increment_cache("counter", 1.0, ExactCacheContext::default())
                            .unwrap();
                    }
                });
            }
        });

        assert_eq!(cache.get_cache("counter"), Ok(Some(80_000.0)));
    }

    #[rstest]
    fn a_trait_write_to_a_live_key_keeps_its_deadline(clock: Clock) {
        let cache = cache::<String>(&clock, 4);
        set_cache(&cache, "key", "first".into(), &ttl(10)).unwrap();
        at(&clock, 105);

        set_cache(&cache, "key", "second".into(), &ttl(10)).unwrap();

        assert_eq!(cache.get_cache("key"), Ok(Some("second".into())));
        assert_eq!(cache.expires_at("key"), Ok(secs(110)));
    }

    #[rstest]
    #[case::its_own(ttl(5), secs(5))]
    #[case::the_default_of_the_cache(ExactCacheContext::default(), secs(60))]
    fn a_context_resolves_to_its_ttl_or(
        clock: Clock,
        #[case] context: ExactCacheContext,
        #[case] expected: Option<Duration>,
    ) {
        assert_eq!(cache::<String>(&clock, 4).get_ttl(&context), expected);
    }

    #[rstest]
    #[case::window_semantics(false)]
    #[case::refresh_ttl_is_ignored(true)]
    #[tokio::test]
    async fn an_increment_keeps_the_window_of_a_live_counter(
        clock: Clock,
        #[case] refresh_ttl: bool,
    ) {
        let cache = cache::<f64>(&clock, 4);

        let first = cache
            .async_increment("counter", 2.0, ttl(10), refresh_ttl)
            .await;
        at(&clock, 105);
        let second = cache
            .async_increment("counter", 3.0, ttl(10), refresh_ttl)
            .await;

        assert_eq!((first, second), (Ok(2.0), Ok(5.0)));
        assert_eq!(cache.expires_at("counter"), Ok(secs(110)));
    }

    #[rstest]
    fn an_expired_counter_restarts_from_zero_with_a_new_window(clock: Clock) {
        let cache = cache::<f64>(&clock, 4);
        cache.increment_cache("counter", 2.0, ttl(10)).unwrap();
        at(&clock, 110);

        let restarted = cache.increment_cache("counter", 1.0, ttl(10));

        assert_eq!(restarted, Ok(1.0));
        assert_eq!(cache.expires_at("counter"), Ok(secs(120)));
    }

    #[rstest]
    fn incrementing_a_held_counter_at_capacity_evicts_nothing(clock: Clock) {
        let cache = cache::<f64>(&clock, 2);
        let increment = |key| cache.increment_cache(key, 1.0, ExactCacheContext::default());
        increment("a").unwrap();
        increment("b").unwrap();

        let totals = (increment("a"), increment("b"));

        assert_eq!(totals, (Ok(2.0), Ok(2.0)));
    }

    #[rstest]
    #[tokio::test]
    async fn a_cache_without_capacity_answers_claims_counters_and_sets_but_keeps_none() {
        let claims = InMemoryCache::<String>::new(Some(0), None);
        let counters = InMemoryCache::<f64>::new(Some(0), None);
        let sets = InMemoryCache::<HashSet<String>>::new(Some(0), None);

        let claimed = claims.claim_cache("key", "first".into(), &[], ExactCacheContext::default());
        let counted = counters.increment_cache("key", 2.0, ExactCacheContext::default());
        let added = sets
            .async_set_cache_sadd("key", vec!["a".into()], None)
            .await;

        assert_eq!(claimed, Ok("first".into()));
        assert_eq!(counted, Ok(2.0));
        assert_eq!(added, Ok(vec!["a".to_owned()]));
        assert_eq!(
            (claims.is_empty(), counters.is_empty(), sets.is_empty()),
            (Ok(true), Ok(true), Ok(true))
        );
    }

    #[rstest]
    #[tokio::test]
    async fn an_increment_pipeline_answers_in_operation_order(clock: Clock) {
        let cache = cache::<f64>(&clock, 3);
        let operation = |key: &str, amount, ttl| IncrementOperation {
            key: key.into(),
            amount,
            ttl: secs(ttl),
        };

        let totals = cache
            .async_increment_pipeline(vec![
                operation("a", 1.0, 10),
                operation("b", 5.0, 30),
                operation("a", 2.0, 20),
            ])
            .await;

        assert_eq!(totals, Ok(vec![1.0, 5.0, 3.0]));
        assert_eq!(cache.expires_at("a"), Ok(secs(110)));
        assert_eq!(cache.expires_at("b"), Ok(secs(130)));
    }

    #[rstest]
    #[tokio::test]
    async fn set_additions_answer_what_was_added_and_store_the_union(clock: Clock) {
        let cache = cache::<HashSet<String>>(&clock, 4);
        let added = vec!["a".to_owned(), "a".into(), "b".into()];

        let first = cache
            .async_set_cache_sadd("members", added.clone(), secs(10))
            .await;
        let second = cache
            .async_set_cache_sadd("members", vec!["c".into()], secs(99))
            .await;

        assert_eq!((first, second), (Ok(added), Ok(vec!["c".to_owned()])));
        assert_eq!(
            cache.get_cache("members"),
            Ok(Some(HashSet::from(["a".into(), "b".into(), "c".into()])))
        );
        assert_eq!(cache.expires_at("members"), Ok(secs(110)));
    }

    #[rstest]
    #[tokio::test]
    async fn an_expired_set_restarts_empty_with_a_new_ttl(clock: Clock) {
        let cache = cache::<HashSet<String>>(&clock, 4);
        cache
            .async_set_cache_sadd("members", vec!["a".into()], secs(10))
            .await
            .unwrap();
        at(&clock, 110);

        cache
            .async_set_cache_sadd("members", vec!["d".into()], None)
            .await
            .unwrap();

        assert_eq!(
            cache.get_cache("members"),
            Ok(Some(HashSet::from(["d".into()])))
        );
        assert_eq!(cache.expires_at("members"), Ok(secs(170)));
    }

    #[rstest]
    #[tokio::test]
    async fn a_set_addition_that_outgrows_the_entry_limit_is_not_stored() {
        let cache = InMemoryCache::<HashSet<String>>::with_clock_and_size_measurement(
            Some(4),
            None,
            Some(2),
            Some(Arc::new(|value: &HashSet<String>| Ok(value.len()))),
            || Duration::ZERO,
        );
        cache
            .async_set_cache_sadd("members", vec!["a".into(), "b".into()], None)
            .await
            .unwrap();

        let added = cache
            .async_set_cache_sadd("members", vec!["c".into()], None)
            .await;

        assert_eq!(added, Ok(vec!["c".to_owned()]));
        assert_eq!(
            cache.get_cache("members"),
            Ok(Some(HashSet::from(["a".into(), "b".into()])))
        );
    }
}
