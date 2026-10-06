mod support;

use std::{
    sync::{
        Arc, Mutex,
        atomic::{AtomicU64, Ordering},
    },
    time::Duration,
};

use litellm_cache::{
    BaseCache, Error, SemanticCacheContext,
    semantic::{SemanticCache, SemanticLookup},
};
use litellm_cache_memory::InMemoryCache;
use litellm_cache_response::{
    CacheControls, CacheEntry, CacheKeyField, CacheKeyInput, CacheKeyParticipation, PendingWrite,
    ResponseCache, ResponseCacheRequest, WriteBuffer, get_cache_key,
};
use redis_test::MockCmd;
use rstest::{fixture, rstest};
use serde_json::{Value, json};
use support::{keyed, memory, redis, request};

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
    request: ResponseCacheRequest,
    #[values(false, true)] asynchronous: bool,
) {
    let backend = Arc::new(SemanticBackend::default());
    let cache = ResponseCache::new(backend.clone());
    let context = SemanticCacheContext {
        messages: Some(json!([{"role": "user", "content": "hello"}])),
        ..Default::default()
    };
    let request = request.with_context(context.clone());
    let response = json!({"answer": 42});
    let now = Duration::from_secs(100);

    let hit = if asynchronous {
        cache
            .async_store(&request, response.clone(), now)
            .await
            .unwrap();
        cache.async_lookup(&request, now).await.unwrap()
    } else {
        cache.store(&request, response.clone(), now).unwrap();
        cache.lookup(&request, now).unwrap()
    };

    assert_eq!(hit, Some(response));
    assert_eq!(
        backend.contexts.lock().unwrap().as_slice(),
        &[context.clone(), context]
    );
}

#[rstest]
#[tokio::test]
async fn batch_store_preserves_semantic_context_without_batch_reads(
    #[values(false, true)] distinct_contexts: bool,
) {
    let backend = Arc::new(SemanticBackend::default());
    let cache = ResponseCache::new(backend.clone());
    let requests = ["first", "second"].map(|key| {
        keyed(key).with_context(SemanticCacheContext {
            messages: Some(json!([{"role": "user", "content": if distinct_contexts { key } else { "shared" }}])),
            ..Default::default()
        })
    });
    let now = Duration::from_secs(100);
    cache
        .async_store_batch(
            requests
                .iter()
                .map(|request| {
                    (
                        request.clone(),
                        json!({"answer": get_cache_key(&request.key)}),
                    )
                })
                .collect(),
            now,
        )
        .await
        .unwrap();

    assert_eq!(
        backend.contexts.lock().unwrap().as_slice(),
        &[requests[0].context.clone(), requests[1].context.clone()]
    );
    for request in requests {
        assert_eq!(
            cache.async_lookup(&request, now).await.unwrap(),
            Some(json!({"answer": get_cache_key(&request.key)}))
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
#[case::fresh_hit(scored(95.0, 0.95), true, Ok(SemanticLookup { value: Some(json!({"answer": 42})), similarity: Some(0.95) }))]
#[case::stale_hit_keeps_the_similarity(
    scored(50.0, 0.95),
    true,
    Ok(SemanticLookup::miss(Some(0.95)))
)]
#[case::miss_keeps_the_similarity(
    Ok(SemanticLookup::miss(Some(0.4))),
    true,
    Ok(SemanticLookup::miss(Some(0.4)))
)]
#[case::no_search(Ok(SemanticLookup::miss(None)), true, Ok(SemanticLookup::miss(None)))]
#[case::disabled_reads_skip_the_backend(scored(95.0, 0.95), false, Ok(SemanticLookup::miss(None)))]
#[case::invalid_entry_is_a_miss(Err(Error::InvalidEntry), true, Ok(SemanticLookup::miss(None)))]
#[case::backend_errors_propagate(Err(Error::Unavailable), true, Err(Error::Unavailable))]
#[tokio::test]
async fn semantic_lookup_applies_freshness_to_the_value_only(
    #[case] backend: Result<SemanticLookup<CacheEntry>, Error>,
    #[case] reads: bool,
    #[case] expected: Result<SemanticLookup<Value>, Error>,
    #[values(false, true)] asynchronous: bool,
    request: ResponseCacheRequest,
) {
    let cache = ResponseCache::new(Arc::new(ScoredBackend(backend)));
    let mut request = request.with_context(SemanticCacheContext::default());
    request.max_age = Some(Duration::from_secs(10));
    request.controls.no_cache = !reads;
    let now = Duration::from_secs(100);

    let lookup = if asynchronous {
        cache.async_lookup_semantic(&request, now).await
    } else {
        cache.lookup_semantic(&request, now)
    };

    assert_eq!(lookup, expected);
}
