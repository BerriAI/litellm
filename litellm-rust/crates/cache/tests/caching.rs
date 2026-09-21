use std::{sync::Mutex, time::Duration};

use litellm_cache::{
    BaseCache, CacheConnectionResult, CacheContext, Error, ExactCacheContext, SemanticCacheContext,
    get_cache,
};

struct TestCache {
    default_ttl: Duration,
    writes: Mutex<Vec<(String, String, ExactCacheContext)>>,
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

    async fn disconnect(&self) -> Result<(), Error> {
        Ok(())
    }

    async fn test_connection(&self) -> Result<CacheConnectionResult, Error> {
        unreachable!()
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
        cache.get_ttl(&ExactCacheContext::default()),
        Some(Duration::from_secs(60))
    );
    assert_eq!(
        cache.get_ttl(&ExactCacheContext {
            ttl: Some(Duration::from_secs(5)),
        }),
        Some(Duration::from_secs(5))
    );
}

#[test]
fn associated_context_preserves_backend_specific_lookup_inputs() {
    let context = SemanticContext {
        ttl: None,
        query: "matching prompt".into(),
    };
    assert_eq!(
        get_cache(&SemanticCache, "shared-key", &context).unwrap(),
        Some("semantic hit".into())
    );
}

#[test]
fn semantic_context_with_ttl_preserves_lookup_inputs() {
    let context = SemanticCacheContext {
        input: Some(serde_json::json!("text")),
        messages: Some(serde_json::json!([{"role": "user", "content": "hi"}])),
        metadata: Some(serde_json::json!({"key": "value"})),
        scope: Some("scope".into()),
        ttl: None,
    };
    let updated = context.with_ttl(Some(Duration::from_secs(30)));
    assert_eq!(updated.ttl(), Some(Duration::from_secs(30)));
    assert_eq!(updated.input, context.input);
    assert_eq!(updated.messages, context.messages);
    assert_eq!(updated.metadata, context.metadata);
    assert_eq!(updated.scope, context.scope);
    assert_eq!(context.with_ttl(None).ttl(), None);
}

#[tokio::test]
async fn default_batch_operations_use_async_writes_and_stop_on_failure() {
    let cache = TestCache {
        default_ttl: Duration::from_secs(60),
        writes: Mutex::default(),
    };
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
