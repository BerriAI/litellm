mod support;

use std::{
    sync::{Arc, Mutex},
    time::Duration,
};

use litellm_cache::{
    BaseCache, Error, SemanticCacheContext,
    semantic::{SemanticCache, SemanticLookup},
};
use litellm_cache_response::{CacheEntry, CacheKey, PendingWrite, ResponseCache};
use rstest::rstest;
use serde_json::{Value, json};
use support::{key, keyed};

#[derive(Default)]
struct SemanticBackend {
    entries: Mutex<Vec<(String, CacheEntry)>>,
    contexts: Mutex<Vec<SemanticCacheContext>>,
}

impl BaseCache for SemanticBackend {
    type Value = CacheEntry;
    type Context = SemanticCacheContext;

    fn get_ttl(&self, _: &Self::Context) -> Option<Duration> {
        None
    }

    fn set_cache(
        &self,
        key: &str,
        value: Self::Value,
        context: &Self::Context,
    ) -> Result<(), Error> {
        self.contexts.lock().unwrap().push(context.clone());
        self.entries.lock().unwrap().push((key.to_owned(), value));
        Ok(())
    }

    fn get_cache(&self, key: &str, context: &Self::Context) -> Result<Option<Self::Value>, Error> {
        self.contexts.lock().unwrap().push(context.clone());
        Ok(self
            .entries
            .lock()
            .unwrap()
            .iter()
            .find(|(entry_key, _)| entry_key == key)
            .map(|(_, entry)| entry.clone()))
    }
}

#[rstest]
#[tokio::test]
async fn semantic_context_reaches_backend_for_store_and_lookup(
    key: CacheKey,
    #[values(false, true)] asynchronous: bool,
) {
    let backend = Arc::new(SemanticBackend::default());
    let cache = ResponseCache::new(backend.clone());
    let context = SemanticCacheContext {
        messages: Some(json!([{"role": "user", "content": "hello"}])),
        ..Default::default()
    };
    let response = json!({"answer": 42});
    let now = Duration::from_secs(100);

    let hit = if asynchronous {
        cache
            .async_store(&key, context.clone(), response.clone(), now)
            .await
            .unwrap();
        cache.async_lookup(&key, &context, None, now).await.unwrap()
    } else {
        cache.store(&key, &context, response.clone(), now).unwrap();
        cache.lookup(&key, &context, None, now).unwrap()
    };

    assert_eq!(hit, Some(response));
    assert_eq!(
        backend.contexts.lock().unwrap().as_slice(),
        &[context.clone(), context]
    );
}

#[rstest]
#[tokio::test]
async fn deferred_writes_preserve_each_semantic_context(
    #[values(false, true)] distinct_contexts: bool,
) {
    let backend = Arc::new(SemanticBackend::default());
    let cache = ResponseCache::new(backend.clone());
    let writes = ["first", "second"].map(|name| PendingWrite {
        key: keyed(name),
        context: SemanticCacheContext {
            messages: Some(json!([{"role": "user", "content": if distinct_contexts { name } else { "shared" }}])),
            ..Default::default()
        },
        response: json!({"answer": name}),
        produced_at: Duration::from_secs(100),
    });
    let expected = writes
        .iter()
        .map(|write| {
            (
                write.key.clone(),
                write.context.clone(),
                write.response.clone(),
            )
        })
        .collect::<Vec<_>>();
    cache.async_store_entries(writes.into()).await.unwrap();

    assert_eq!(
        backend.contexts.lock().unwrap().as_slice(),
        &[expected[0].1.clone(), expected[1].1.clone()]
    );
    for (key, context, response) in expected {
        assert_eq!(
            cache
                .async_lookup(&key, &context, None, Duration::from_secs(100))
                .await
                .unwrap(),
            Some(response)
        );
    }
}

struct ScoredBackend(Result<SemanticLookup<CacheEntry>, Error>);

impl BaseCache for ScoredBackend {
    type Value = CacheEntry;
    type Context = SemanticCacheContext;

    fn get_ttl(&self, _: &Self::Context) -> Option<Duration> {
        None
    }

    fn set_cache(&self, _: &str, _: Self::Value, _: &Self::Context) -> Result<(), Error> {
        Ok(())
    }

    fn get_cache(&self, key: &str, context: &Self::Context) -> Result<Option<Self::Value>, Error> {
        self.get_cache_with_similarity(key, context)
            .map(|lookup| lookup.value)
    }
}

impl SemanticCache for ScoredBackend {
    fn get_cache_with_similarity(
        &self,
        _: &str,
        _: &Self::Context,
    ) -> Result<SemanticLookup<Self::Value>, Error> {
        self.0.clone()
    }

    async fn async_get_cache_with_similarity(
        &self,
        key: &str,
        context: &Self::Context,
    ) -> Result<SemanticLookup<Self::Value>, Error> {
        self.get_cache_with_similarity(key, context)
    }
}

fn scored(timestamp: f64, similarity: f64) -> Result<SemanticLookup<CacheEntry>, Error> {
    Ok(SemanticLookup {
        value: Some(CacheEntry {
            timestamp: Some(timestamp),
            response: json!({"answer": 42}),
        }),
        similarity: Some(similarity),
    })
}

#[rstest]
#[case::fresh_hit(scored(95.0, 0.95), Ok(SemanticLookup { value: Some(json!({"answer": 42})), similarity: Some(0.95) }))]
#[case::stale_hit_keeps_the_similarity(scored(50.0, 0.95), Ok(SemanticLookup::miss(Some(0.95))))]
#[case::miss_keeps_the_similarity(
    Ok(SemanticLookup::miss(Some(0.4))),
    Ok(SemanticLookup::miss(Some(0.4)))
)]
#[case::no_search(Ok(SemanticLookup::miss(None)), Ok(SemanticLookup::miss(None)))]
#[case::invalid_entry_is_a_miss(Err(Error::InvalidEntry), Ok(SemanticLookup::miss(None)))]
#[case::backend_errors_propagate(Err(Error::Unavailable), Err(Error::Unavailable))]
#[tokio::test]
async fn semantic_lookup_applies_freshness_to_the_value_only(
    #[case] backend: Result<SemanticLookup<CacheEntry>, Error>,
    #[case] expected: Result<SemanticLookup<Value>, Error>,
    #[values(false, true)] asynchronous: bool,
    key: CacheKey,
) {
    let cache = ResponseCache::new(Arc::new(ScoredBackend(backend)));
    let context = SemanticCacheContext::default();
    let max_age = Some(Duration::from_secs(10));
    let now = Duration::from_secs(100);

    let lookup = if asynchronous {
        cache
            .async_lookup_semantic(&key, &context, max_age, now)
            .await
    } else {
        cache.lookup_semantic(&key, &context, max_age, now)
    };

    assert_eq!(lookup, expected);
}
