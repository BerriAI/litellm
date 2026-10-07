use std::{
    sync::{Arc, Mutex},
    time::Duration,
};

use litellm_cache::{
    BaseCache, BatchCache, CacheConnectionResult, ClaimCache, CounterCache, DeleteCache, DualCache,
    Error, ExactCacheContext, FlushCache, ReadPolicy, RemoteFailurePolicy, WritePolicy,
};

struct TestCache<V> {
    value: Mutex<Option<V>>,
    fail: bool,
}

impl<V> TestCache<V> {
    fn new(value: Option<V>, fail: bool) -> Self {
        Self {
            value: Mutex::new(value),
            fail,
        }
    }
}

impl<V> BaseCache for TestCache<V>
where
    V: Clone + Send + Sync + 'static,
{
    type Value = V;
    type Context = ExactCacheContext;

    fn get_ttl(&self, context: &Self::Context) -> Option<Duration> {
        context.ttl.or(Some(Duration::from_secs(60)))
    }

    fn set_cache(&self, _: &str, value: V, _: &ExactCacheContext) -> Result<(), Error> {
        *self.value.lock().unwrap() = Some(value);
        Ok(())
    }

    fn get_cache(&self, _: &str, _: &ExactCacheContext) -> Result<Option<V>, Error> {
        Ok(self.value.lock().unwrap().clone())
    }

    async fn disconnect(&self) -> Result<(), Error> {
        Ok(())
    }

    async fn test_connection(&self) -> Result<CacheConnectionResult, Error> {
        unreachable!()
    }
}

impl<V> BatchCache for TestCache<V> where V: Clone + Send + Sync + 'static {}

impl<V> DeleteCache for TestCache<V>
where
    V: Clone + Send + Sync + 'static,
{
    fn delete_cache(&self, _: &str) -> Result<(), Error> {
        *self.value.lock().unwrap() = None;
        Ok(())
    }
}

impl<V> FlushCache for TestCache<V>
where
    V: Clone + Send + Sync + 'static,
{
    fn flush_cache(&self) -> Result<(), Error> {
        *self.value.lock().unwrap() = None;
        Ok(())
    }
}

impl CounterCache for TestCache<f64> {
    fn increment_cache(&self, _: &str, amount: f64, _: ExactCacheContext) -> Result<f64, Error> {
        if self.fail {
            return Err(Error::Unavailable);
        }
        let mut value = self.value.lock().unwrap();
        let incremented = value.unwrap_or_default() + amount;
        *value = Some(incremented);
        Ok(incremented)
    }
}

impl<V> ClaimCache for TestCache<V>
where
    V: Clone + PartialEq + Send + Sync + 'static,
{
    fn claim_cache(
        &self,
        _: &str,
        candidate: V,
        eligible: &[V],
        _: ExactCacheContext,
    ) -> Result<V, Error> {
        if self.fail {
            return Err(Error::Unavailable);
        }
        let mut value = self.value.lock().unwrap();
        let winner = match value.as_ref() {
            Some(existing) if eligible.is_empty() || eligible.contains(existing) => {
                existing.clone()
            }
            _ => candidate,
        };
        *value = Some(winner.clone());
        Ok(winner)
    }
}

#[test]
fn failed_l2_increment_leaves_l1_unchanged() {
    let l1 = Arc::new(TestCache::new(Some(10.0), false));
    let cache = DualCache::new(l1.clone(), Arc::new(TestCache::new(Some(20.0), true)));

    assert_eq!(
        cache.increment_cache("counter", 2.0, ExactCacheContext::default()),
        Err(Error::Unavailable)
    );
    assert_eq!(
        l1.get_cache("counter", &ExactCacheContext::default())
            .unwrap(),
        Some(10.0)
    );
}

#[test]
fn claim_uses_l1_fallback_without_overwriting_an_eligible_winner() {
    let l1 = Arc::new(TestCache::new(Some("first".to_string()), false));
    let cache = DualCache::new(l1, Arc::new(TestCache::new(None, true)))
        .with_remote_failure_policy(RemoteFailurePolicy::UseLocal);

    assert_eq!(
        cache
            .claim_cache(
                "affinity",
                "second".into(),
                &["first".into(), "second".into()],
                ExactCacheContext {
                    ttl: Some(Duration::from_secs(60)),
                },
            )
            .unwrap(),
        "first"
    );
}

struct SyncPanics(TestCache<String>);

impl BaseCache for SyncPanics {
    type Value = String;
    type Context = ExactCacheContext;

    fn get_ttl(&self, context: &Self::Context) -> Option<Duration> {
        self.0.get_ttl(context)
    }

    fn set_cache(&self, _: &str, _: String, _: &ExactCacheContext) -> Result<(), Error> {
        panic!("sync L2 write on an async path")
    }

    fn get_cache(&self, _: &str, _: &ExactCacheContext) -> Result<Option<String>, Error> {
        panic!("sync L2 read on an async path")
    }

    async fn async_set_cache(
        &self,
        key: &str,
        value: String,
        context: ExactCacheContext,
    ) -> Result<(), Error> {
        self.0.set_cache(key, value, &context)
    }

    async fn async_get_cache(
        &self,
        key: &str,
        context: &ExactCacheContext,
    ) -> Result<Option<String>, Error> {
        self.0.get_cache(key, context)
    }

    async fn async_set_cache_pipeline(
        &self,
        cache_list: Vec<(String, String)>,
        context: ExactCacheContext,
    ) -> Result<(), Error> {
        for (key, value) in cache_list {
            self.0.set_cache(&key, value, &context)?;
        }
        Ok(())
    }

    async fn disconnect(&self) -> Result<(), Error> {
        Ok(())
    }

    async fn test_connection(&self) -> Result<CacheConnectionResult, Error> {
        unreachable!()
    }
}

impl BatchCache for SyncPanics {
    async fn async_batch_get_cache(
        &self,
        keys: Vec<String>,
        context: ExactCacheContext,
    ) -> Result<Vec<litellm_cache::BatchEntry<String>>, Error> {
        assert_eq!(keys, ["missing"]);
        Ok(vec![match self.0.get_cache("missing", &context)? {
            Some(value) => litellm_cache::BatchEntry::Hit(value),
            None => litellm_cache::BatchEntry::Miss,
        }])
    }
}

impl DeleteCache for SyncPanics {
    fn delete_cache(&self, _: &str) -> Result<(), Error> {
        panic!("sync L2 delete on an async path")
    }

    async fn async_delete_cache(&self, key: &str) -> Result<(), Error> {
        self.0.delete_cache(key)
    }
}

impl FlushCache for SyncPanics {
    fn flush_cache(&self) -> Result<(), Error> {
        panic!("sync L2 flush on an async path")
    }
}

#[tokio::test]
async fn async_operations_use_the_async_l2_methods() {
    let l1 = Arc::new(TestCache::new(None, false));
    let cache = DualCache::new(
        l1.clone(),
        Arc::new(SyncPanics(TestCache::new(
            Some("remote".to_string()),
            false,
        ))),
    );
    let context = ExactCacheContext::default();

    assert_eq!(
        cache.async_get_cache("missing", &context).await.unwrap(),
        Some("remote".into())
    );
    assert_eq!(
        l1.get_cache("missing", &context).unwrap(),
        Some("remote".into())
    );

    l1.delete_cache("missing").unwrap();
    assert_eq!(
        cache
            .async_batch_get_cache(vec!["missing".into()], context.clone())
            .await
            .unwrap(),
        [litellm_cache::BatchEntry::Hit("remote".to_string())]
    );
    cache
        .async_set_cache("missing", "written".into(), context.clone())
        .await
        .unwrap();
    cache
        .async_set_cache_pipeline(vec![("missing".into(), "piped".into())], context.clone())
        .await
        .unwrap();
    cache.async_delete_cache("missing").await.unwrap();
    assert_eq!(
        cache.async_get_cache("missing", &context).await.unwrap(),
        None
    );
}

struct Unavailable;

impl BaseCache for Unavailable {
    type Value = String;
    type Context = ExactCacheContext;

    fn get_ttl(&self, context: &Self::Context) -> Option<Duration> {
        context.ttl
    }

    fn set_cache(&self, _: &str, _: String, _: &ExactCacheContext) -> Result<(), Error> {
        Err(Error::Unavailable)
    }

    fn get_cache(&self, _: &str, _: &ExactCacheContext) -> Result<Option<String>, Error> {
        Err(Error::Unavailable)
    }

    async fn disconnect(&self) -> Result<(), Error> {
        Ok(())
    }

    async fn test_connection(&self) -> Result<CacheConnectionResult, Error> {
        unreachable!()
    }
}

impl BatchCache for Unavailable {}

impl DeleteCache for Unavailable {
    fn delete_cache(&self, _: &str) -> Result<(), Error> {
        Err(Error::Unavailable)
    }
}

impl FlushCache for Unavailable {
    fn flush_cache(&self) -> Result<(), Error> {
        Err(Error::Unavailable)
    }
}

impl ClaimCache for Unavailable {
    fn claim_cache(
        &self,
        _: &str,
        _: String,
        _: &[String],
        _: ExactCacheContext,
    ) -> Result<String, Error> {
        Err(Error::InvalidEntry)
    }
}

#[test]
fn remote_failure_policy_selects_propagation_or_the_local_tier() {
    let context = ExactCacheContext::default();
    let strict = DualCache::new(Arc::new(TestCache::new(None, false)), Arc::new(Unavailable));
    assert_eq!(
        strict.set_cache("key", "value".into(), &context),
        Err(Error::Unavailable)
    );
    assert_eq!(strict.get_cache("key", &context), Err(Error::Unavailable));

    let l1 = Arc::new(TestCache::new(None, false));
    let degraded = DualCache::new(l1.clone(), Arc::new(Unavailable))
        .with_remote_failure_policy(RemoteFailurePolicy::UseLocal);
    assert_eq!(degraded.get_cache("key", &context), Ok(None));
    degraded.set_cache("key", "value".into(), &context).unwrap();
    assert_eq!(
        degraded.get_cache("key", &context),
        Ok(Some("value".into()))
    );
    degraded.delete_cache("key").unwrap();
    assert_eq!(l1.get_cache("key", &context), Ok(None));
}

#[test]
fn claim_fallback_does_not_hide_non_availability_errors() {
    let cache = DualCache::new(
        Arc::new(TestCache::new(Some("first".to_string()), false)),
        Arc::new(Unavailable),
    )
    .with_remote_failure_policy(RemoteFailurePolicy::UseLocal);
    assert_eq!(
        cache.claim_cache(
            "affinity",
            "second".into(),
            &[],
            ExactCacheContext::default()
        ),
        Err(Error::InvalidEntry)
    );
}

#[test]
fn local_only_policies_never_touch_l2() {
    let l2 = Arc::new(TestCache::new(Some("remote".to_string()), false));
    let cache = DualCache::new(Arc::new(TestCache::new(None, false)), l2.clone())
        .with_read_policy(ReadPolicy::LocalOnly)
        .with_write_policy(WritePolicy::LocalOnly);
    let context = ExactCacheContext::default();

    assert_eq!(cache.get_cache("key", &context), Ok(None));
    cache.set_cache("key", "local".into(), &context).unwrap();
    assert_eq!(l2.get_cache("key", &context), Ok(Some("remote".into())));
}
