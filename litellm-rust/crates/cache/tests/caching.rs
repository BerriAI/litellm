use std::{sync::Mutex, time::Duration};

use litellm_cache::{
    BaseCache, CacheContext, CounterCache, Error, ExactCacheContext, IncrementOperation,
    SemanticCacheContext, get_cache,
};
use rstest::{fixture, rstest};
use serde_json::json;

struct TestCache {
    default_ttl: Duration,
    writes: Mutex<Vec<(String, String, ExactCacheContext)>>,
}

#[fixture]
fn cache() -> TestCache {
    TestCache {
        default_ttl: Duration::from_secs(60),
        writes: Mutex::default(),
    }
}

#[derive(Clone)]
struct SemanticContext {
    ttl: Option<Duration>,
    query: String,
}

impl CacheContext for SemanticContext {
    fn ttl(&self) -> Option<Duration> {
        self.ttl
    }

    fn with_ttl(&self, ttl: Option<Duration>) -> Self {
        Self {
            ttl,
            query: self.query.clone(),
        }
    }
}

struct SemanticCache;

impl BaseCache for SemanticCache {
    type Value = String;
    type Context = SemanticContext;

    fn get_ttl(&self, context: &Self::Context) -> Option<Duration> {
        context.ttl
    }

    fn set_cache(&self, _: &str, _: Self::Value, _: &Self::Context) -> Result<(), Error> {
        Ok(())
    }

    fn get_cache(&self, _: &str, context: &Self::Context) -> Result<Option<Self::Value>, Error> {
        Ok((context.query == "matching prompt").then(|| "semantic hit".into()))
    }
}

impl BaseCache for TestCache {
    type Value = String;
    type Context = ExactCacheContext;

    fn get_ttl(&self, context: &Self::Context) -> Option<Duration> {
        context.ttl.or(Some(self.default_ttl))
    }

    fn set_cache(&self, _: &str, _: Self::Value, _: &ExactCacheContext) -> Result<(), Error> {
        Err(Error::Unavailable)
    }

    async fn async_set_cache(
        &self,
        key: &str,
        value: Self::Value,
        context: ExactCacheContext,
    ) -> Result<(), Error> {
        if key == "unavailable" {
            return Err(Error::Unavailable);
        }
        self.writes
            .lock()
            .unwrap()
            .push((key.into(), value, context));
        Ok(())
    }

    fn get_cache(&self, _: &str, _: &ExactCacheContext) -> Result<Option<Self::Value>, Error> {
        Ok(None)
    }
}

/// Records every `async_increment` so the default pipeline's calls are observable.
#[derive(Default)]
struct RecordingCounter {
    increments: Mutex<Vec<(String, f64, ExactCacheContext, bool)>>,
    total: Mutex<f64>,
}

impl BaseCache for RecordingCounter {
    type Value = f64;
    type Context = ExactCacheContext;

    fn get_ttl(&self, context: &Self::Context) -> Option<Duration> {
        context.ttl
    }

    fn set_cache(&self, _: &str, _: f64, _: &ExactCacheContext) -> Result<(), Error> {
        Ok(())
    }

    fn get_cache(&self, _: &str, _: &ExactCacheContext) -> Result<Option<f64>, Error> {
        Ok(None)
    }
}

impl CounterCache for RecordingCounter {
    fn increment_cache(&self, key: &str, amount: f64, _: ExactCacheContext) -> Result<f64, Error> {
        if key == "unavailable" {
            return Err(Error::Unavailable);
        }
        let mut total = self.total.lock().unwrap();
        *total += amount;
        Ok(*total)
    }

    async fn async_increment(
        &self,
        key: &str,
        amount: f64,
        context: ExactCacheContext,
        refresh_ttl: bool,
    ) -> Result<f64, Error> {
        self.increments
            .lock()
            .unwrap()
            .push((key.into(), amount, context.clone(), refresh_ttl));
        self.increment_cache(key, amount, context)
    }
}

fn operation(key: &str, amount: f64, ttl: Option<u64>) -> IncrementOperation {
    IncrementOperation {
        key: key.into(),
        amount,
        ttl: ttl.map(Duration::from_secs),
    }
}

#[rstest]
#[case::default_ttl(None, Some(60))]
#[case::per_call_override(Some(5), Some(5))]
fn ttl_uses_default_and_allows_per_call_override(
    cache: TestCache,
    #[case] ttl: Option<u64>,
    #[case] expected: Option<u64>,
) {
    assert_eq!(
        cache.get_ttl(&ExactCacheContext {
            ttl: ttl.map(Duration::from_secs),
        }),
        expected.map(Duration::from_secs)
    );
}

#[rstest]
#[case::matching("matching prompt", Some("semantic hit"))]
#[case::other("other prompt", None)]
fn associated_context_preserves_backend_specific_lookup_inputs(
    #[case] query: &str,
    #[case] expected: Option<&str>,
) {
    let context = SemanticContext {
        ttl: None,
        query: query.into(),
    };
    assert_eq!(
        get_cache(&SemanticCache, "shared-key", &context).unwrap(),
        expected.map(String::from)
    );
}

#[rstest]
#[case::set(None, Some(30))]
#[case::replaced(Some(10), Some(20))]
#[case::cleared(Some(10), None)]
fn semantic_context_with_ttl_only_replaces_ttl(
    #[case] initial: Option<u64>,
    #[case] updated: Option<u64>,
) {
    let context = SemanticCacheContext {
        input: Some(json!({"input": "hello"})),
        messages: Some(json!([{"role": "user", "content": "hello"}])),
        metadata: Some(json!({"tenant": "team"})),
        scope: Some("scope".into()),
        ttl: initial.map(Duration::from_secs),
    };
    let result = context.with_ttl(updated.map(Duration::from_secs));
    assert_eq!(result.ttl(), updated.map(Duration::from_secs));
    assert_eq!(result.input, context.input);
    assert_eq!(result.messages, context.messages);
    assert_eq!(result.metadata, context.metadata);
    assert_eq!(result.scope, context.scope);
}

#[rstest]
#[tokio::test]
async fn default_batch_operations_use_async_writes_and_stop_on_failure(cache: TestCache) {
    let entry = String::from("cached");
    let context = ExactCacheContext {
        ttl: Some(Duration::from_secs(5)),
    };
    cache
        .batch_cache_write("single", entry.clone(), context.clone())
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
                context.clone(),
            )
            .await,
        Err(Error::Unavailable)
    );
    assert_eq!(
        *cache.writes.lock().unwrap(),
        vec![
            ("single".into(), entry.clone(), context.clone()),
            ("first".into(), entry, context),
        ]
    );
}

#[rstest]
#[tokio::test]
async fn default_async_increment_delegates_to_the_sync_increment() {
    struct SyncOnly;

    impl BaseCache for SyncOnly {
        type Value = f64;
        type Context = ExactCacheContext;

        fn get_ttl(&self, context: &Self::Context) -> Option<Duration> {
            context.ttl
        }

        fn set_cache(&self, _: &str, _: f64, _: &ExactCacheContext) -> Result<(), Error> {
            Ok(())
        }

        fn get_cache(&self, _: &str, _: &ExactCacheContext) -> Result<Option<f64>, Error> {
            Ok(None)
        }
    }

    impl CounterCache for SyncOnly {
        fn increment_cache(
            &self,
            _: &str,
            amount: f64,
            _: ExactCacheContext,
        ) -> Result<f64, Error> {
            Ok(amount * 10.0)
        }
    }

    for refresh_ttl in [false, true] {
        assert_eq!(
            SyncOnly
                .async_increment("key", 2.0, ExactCacheContext::default(), refresh_ttl)
                .await,
            Ok(20.0)
        );
    }
}

#[rstest]
#[case::empty(Vec::new(), Vec::new())]
#[case::one(vec![operation("a", 1.0, Some(10))], vec![1.0])]
#[case::in_order(
    vec![operation("a", 1.0, Some(10)), operation("b", 2.5, None), operation("a", -0.5, Some(20))],
    vec![1.0, 3.5, 3.0],
)]
#[tokio::test]
async fn default_increment_pipeline_increments_each_operation_in_order(
    #[case] operations: Vec<IncrementOperation>,
    #[case] expected: Vec<f64>,
) {
    let cache = RecordingCounter::default();
    assert_eq!(
        cache.async_increment_pipeline(operations.clone()).await,
        Ok(expected)
    );
    assert_eq!(
        *cache.increments.lock().unwrap(),
        operations
            .into_iter()
            .map(|operation| (
                operation.key,
                operation.amount,
                ExactCacheContext { ttl: operation.ttl },
                false,
            ))
            .collect::<Vec<_>>()
    );
}

#[rstest]
#[tokio::test]
async fn default_increment_pipeline_stops_at_the_first_failure() {
    let cache = RecordingCounter::default();
    assert_eq!(
        cache
            .async_increment_pipeline(vec![
                operation("a", 1.0, None),
                operation("unavailable", 1.0, None),
                operation("skipped", 1.0, None),
            ])
            .await,
        Err(Error::Unavailable)
    );
    let keys = cache
        .increments
        .lock()
        .unwrap()
        .iter()
        .map(|(key, ..)| key.clone())
        .collect::<Vec<_>>();
    assert_eq!(keys, ["a", "unavailable"]);
}
