use std::{
    sync::{Arc, Mutex},
    time::Duration,
};

use litellm_cache::{
    BaseCache, CacheConnectionResult, CacheKwargs, ClaimCache, CounterCache, Error,
    dual::{DualCache, ReadPolicy, RemoteFailurePolicy, WritePolicy},
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

    fn set_cache(&self, _: &str, value: V, _: CacheKwargs) -> Result<(), Error> {
        *self.value.lock().unwrap() = Some(value);
        Ok(())
    }

    fn get_cache(&self, _: &str, _: &CacheKwargs) -> Result<Option<V>, Error> {
        Ok(self.value.lock().unwrap().clone())
    }

    fn delete_cache(&self, _: &str) -> Result<(), Error> {
        *self.value.lock().unwrap() = None;
        Ok(())
    }

    fn flush_cache(&self) -> Result<(), Error> {
        *self.value.lock().unwrap() = None;
        Ok(())
    }

    async fn disconnect(&self) -> Result<(), Error> {
        Ok(())
    }

    async fn test_connection(&self) -> Result<CacheConnectionResult, Error> {
        unreachable!()
    }
}

impl CounterCache for TestCache<f64> {
    fn increment_cache(&self, _: &str, amount: f64, _: CacheKwargs) -> Result<f64, Error> {
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
        _: CacheKwargs,
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
        cache.increment_cache("counter", 2.0, CacheKwargs::default()),
        Err(Error::Unavailable)
    );
    assert_eq!(
        l1.get_cache("counter", &CacheKwargs::default()).unwrap(),
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
                CacheKwargs {
                    ttl: Some(Duration::from_secs(60)),
                    ..Default::default()
                },
            )
            .unwrap(),
        "first"
    );
}

struct SyncPanics(TestCache<String>);

impl BaseCache for SyncPanics {
    type Value = String;

    fn set_cache(&self, _: &str, _: String, _: CacheKwargs) -> Result<(), Error> {
        panic!("sync L2 write on an async path")
    }

    fn get_cache(&self, _: &str, _: &CacheKwargs) -> Result<Option<String>, Error> {
        panic!("sync L2 read on an async path")
    }

    async fn async_set_cache(
        &self,
        key: &str,
        value: String,
        kwargs: CacheKwargs,
    ) -> Result<(), Error> {
        self.0.set_cache(key, value, kwargs)
    }

    async fn async_get_cache(
        &self,
        key: &str,
        kwargs: &CacheKwargs,
    ) -> Result<Option<String>, Error> {
        self.0.get_cache(key, kwargs)
    }

    async fn async_get_cache_batch(
        &self,
        keys: Vec<String>,
        kwargs: CacheKwargs,
    ) -> Result<Vec<litellm_cache::BatchEntry<String>>, Error> {
        assert_eq!(keys, ["missing"]);
        Ok(vec![match self.0.get_cache("missing", &kwargs)? {
            Some(value) => litellm_cache::BatchEntry::Hit(value),
            None => litellm_cache::BatchEntry::Miss,
        }])
    }

    async fn async_set_cache_pipeline(
        &self,
        cache_list: Vec<(String, String)>,
        kwargs: CacheKwargs,
    ) -> Result<(), Error> {
        for (key, value) in cache_list {
            self.0.set_cache(&key, value, kwargs.clone())?;
        }
        Ok(())
    }

    fn delete_cache(&self, _: &str) -> Result<(), Error> {
        panic!("sync L2 delete on an async path")
    }

    async fn async_delete_cache(&self, key: &str) -> Result<(), Error> {
        self.0.delete_cache(key)
    }

    fn flush_cache(&self) -> Result<(), Error> {
        panic!("sync L2 flush on an async path")
    }

    async fn disconnect(&self) -> Result<(), Error> {
        Ok(())
    }

    async fn test_connection(&self) -> Result<CacheConnectionResult, Error> {
        unreachable!()
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
    let kwargs = CacheKwargs::default();

    assert_eq!(
        cache.async_get_cache("missing", &kwargs).await.unwrap(),
        Some("remote".into())
    );
    assert_eq!(
        l1.get_cache("missing", &kwargs).unwrap(),
        Some("remote".into())
    );

    l1.delete_cache("missing").unwrap();
    assert_eq!(
        cache
            .async_get_cache_batch(vec!["missing".into()], kwargs.clone())
            .await
            .unwrap(),
        [litellm_cache::BatchEntry::Hit("remote".to_string())]
    );
    cache
        .async_set_cache("missing", "written".into(), kwargs.clone())
        .await
        .unwrap();
    cache
        .async_set_cache_pipeline(vec![("missing".into(), "piped".into())], kwargs.clone())
        .await
        .unwrap();
    cache.async_delete_cache("missing").await.unwrap();
    assert_eq!(
        cache.async_get_cache("missing", &kwargs).await.unwrap(),
        None
    );
}

struct Unavailable;

impl BaseCache for Unavailable {
    type Value = String;

    fn set_cache(&self, _: &str, _: String, _: CacheKwargs) -> Result<(), Error> {
        Err(Error::Unavailable)
    }

    fn get_cache(&self, _: &str, _: &CacheKwargs) -> Result<Option<String>, Error> {
        Err(Error::Unavailable)
    }

    fn delete_cache(&self, _: &str) -> Result<(), Error> {
        Err(Error::Unavailable)
    }

    fn flush_cache(&self) -> Result<(), Error> {
        Err(Error::Unavailable)
    }

    async fn disconnect(&self) -> Result<(), Error> {
        Ok(())
    }

    async fn test_connection(&self) -> Result<CacheConnectionResult, Error> {
        unreachable!()
    }
}

impl ClaimCache for Unavailable {
    fn claim_cache(
        &self,
        _: &str,
        _: String,
        _: &[String],
        _: CacheKwargs,
    ) -> Result<String, Error> {
        Err(Error::InvalidEntry)
    }
}

#[test]
fn remote_failure_policy_selects_propagation_or_the_local_tier() {
    let kwargs = CacheKwargs::default();
    let strict = DualCache::new(Arc::new(TestCache::new(None, false)), Arc::new(Unavailable));
    assert_eq!(
        strict.set_cache("key", "value".into(), kwargs.clone()),
        Err(Error::Unavailable)
    );
    assert_eq!(strict.get_cache("key", &kwargs), Err(Error::Unavailable));

    let l1 = Arc::new(TestCache::new(None, false));
    let degraded = DualCache::new(l1.clone(), Arc::new(Unavailable))
        .with_remote_failure_policy(RemoteFailurePolicy::UseLocal);
    assert_eq!(degraded.get_cache("key", &kwargs), Ok(None));
    degraded
        .set_cache("key", "value".into(), kwargs.clone())
        .unwrap();
    assert_eq!(degraded.get_cache("key", &kwargs), Ok(Some("value".into())));
    degraded.delete_cache("key").unwrap();
    assert_eq!(l1.get_cache("key", &kwargs), Ok(None));
}

#[test]
fn claim_fallback_does_not_hide_non_availability_errors() {
    let cache = DualCache::new(
        Arc::new(TestCache::new(Some("first".to_string()), false)),
        Arc::new(Unavailable),
    )
    .with_remote_failure_policy(RemoteFailurePolicy::UseLocal);
    assert_eq!(
        cache.claim_cache("affinity", "second".into(), &[], CacheKwargs::default()),
        Err(Error::InvalidEntry)
    );
}

#[test]
fn local_only_policies_never_touch_l2() {
    let l2 = Arc::new(TestCache::new(Some("remote".to_string()), false));
    let cache = DualCache::new(Arc::new(TestCache::new(None, false)), l2.clone())
        .with_read_policy(ReadPolicy::LocalOnly)
        .with_write_policy(WritePolicy::LocalOnly);
    let kwargs = CacheKwargs::default();

    assert_eq!(cache.get_cache("key", &kwargs), Ok(None));
    cache
        .set_cache("key", "local".into(), kwargs.clone())
        .unwrap();
    assert_eq!(l2.get_cache("key", &kwargs), Ok(Some("remote".into())));
}
