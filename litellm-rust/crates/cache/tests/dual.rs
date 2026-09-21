use std::{
    sync::{Arc, Mutex},
    time::Duration,
};

use litellm_cache::{
    BaseCache, CacheConnectionResult, CacheKwargs, ClaimCache, CounterCache, Error, dual::DualCache,
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
    let cache = DualCache::new(l1, Arc::new(TestCache::new(None, true)));

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
