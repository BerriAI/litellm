use litellm_cache::{BaseCache, CacheConnectionResult, CacheKwargs, Error};
use std::{sync::Mutex, time::Duration};

struct TestCache {
    default_ttl: Duration,
    writes: Mutex<Vec<(String, String, CacheKwargs)>>,
}

impl BaseCache for TestCache {
    type Value = String;

    fn default_ttl(&self) -> Duration {
        self.default_ttl
    }

    fn set_cache(&self, _: &str, _: Self::Value, _: CacheKwargs) -> Result<(), Error> {
        Err(Error::Unavailable)
    }

    async fn async_set_cache(
        &self,
        key: &str,
        value: Self::Value,
        kwargs: CacheKwargs,
    ) -> Result<(), Error> {
        if key == "unavailable" {
            return Err(Error::Unavailable);
        }
        self.writes
            .lock()
            .unwrap()
            .push((key.into(), value, kwargs));
        Ok(())
    }

    fn get_cache(&self, _: &str, _: &CacheKwargs) -> Result<Option<Self::Value>, Error> {
        Ok(None)
    }

    fn delete_cache(&self, _: &str) -> Result<(), Error> {
        Ok(())
    }

    fn flush_cache(&self) -> Result<(), Error> {
        Ok(())
    }

    async fn disconnect(&self) -> Result<(), Error> {
        Ok(())
    }

    async fn test_connection(&self) -> Result<CacheConnectionResult, Error> {
        unreachable!()
    }
}

#[test]
fn ttl_uses_default_and_allows_per_call_override() {
    let cache = TestCache {
        default_ttl: Duration::from_secs(60),
        writes: Mutex::default(),
    };
    assert_eq!(
        cache.get_ttl(&CacheKwargs::default()),
        Duration::from_secs(60)
    );
    assert_eq!(
        cache.get_ttl(&CacheKwargs {
            ttl: Some(Duration::from_secs(5)),
            ..Default::default()
        }),
        Duration::from_secs(5)
    );
}

#[tokio::test]
async fn default_batch_operations_use_async_writes_and_stop_on_failure() {
    let cache = TestCache {
        default_ttl: Duration::from_secs(60),
        writes: Mutex::default(),
    };
    let entry = String::from("cached");
    let kwargs = CacheKwargs {
        ttl: Some(Duration::from_secs(5)),
        ..Default::default()
    };
    cache
        .batch_cache_write("single", entry.clone(), kwargs.clone())
        .await
        .unwrap();
    assert_eq!(
        cache
            .async_set_cache_pipeline(
                vec![
                    ("first".into(), entry.clone()),
                    ("unavailable".into(), entry.clone()),
                    ("skipped".into(), entry.clone()),
                ],
                kwargs.clone(),
            )
            .await,
        Err(Error::Unavailable)
    );
    assert_eq!(
        *cache.writes.lock().unwrap(),
        vec![
            ("single".into(), entry.clone(), kwargs.clone()),
            ("first".into(), entry, kwargs),
        ]
    );
}
