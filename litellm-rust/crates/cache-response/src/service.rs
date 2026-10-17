use std::{future::Future, pin::Pin, sync::Arc, time::Duration};

use litellm_cache::{BaseCache, Error, ExactCacheContext};
use serde_json::Value;

use crate::{
    CacheControls, CacheEntry, CacheKeyField, CacheKeyInput, ResponseCache, ResponseCacheRequest,
    cache_key,
};

type CacheFuture<'a, T> = Pin<Box<dyn Future<Output = Result<T, Error>> + Send + 'a>>;

#[derive(Clone)]
pub struct ResponseCacheConfig {
    pub namespace: String,
    pub max_entry_bytes: usize,
}

impl Default for ResponseCacheConfig {
    fn default() -> Self {
        Self {
            namespace: String::new(),
            max_entry_bytes: usize::MAX,
        }
    }
}

#[derive(Debug, PartialEq)]
pub struct CacheHit {
    pub value: Value,
    pub key: String,
}

pub trait ResponseCacheService: Send + Sync {
    fn prepare(
        self: Arc<Self>,
        options: CacheOptions,
        surface: &str,
        input: &dyn Fn() -> Value,
    ) -> PreparedCache;
}

pub struct PreparedCache {
    pub policy: CachePolicy,
    pub max_entry_bytes: usize,
    pub call: Arc<dyn ResponseCacheCall>,
}

pub trait ResponseCacheCall: Send + Sync {
    fn lookup<'a>(&'a self, now: Duration) -> CacheFuture<'a, Option<CacheHit>>;

    fn store<'a>(&'a self, response: Value, now: Duration) -> CacheFuture<'a, ()>;
}

impl<B> ResponseCacheService for ResponseCache<B>
where
    B: BaseCache<Value = CacheEntry, Context = ExactCacheContext> + 'static,
{
    fn prepare(
        self: Arc<Self>,
        options: CacheOptions,
        surface: &str,
        input: &dyn Fn() -> Value,
    ) -> PreparedCache {
        let policy = options.policy;
        let request = options.request(&self.config().namespace, surface, input());
        PreparedCache {
            policy,
            max_entry_bytes: self.config().max_entry_bytes,
            call: Arc::new(NativeCacheCall {
                key: cache_key(&request.key),
                request,
                service: self,
            }),
        }
    }
}

struct NativeCacheCall<B: BaseCache<Value = CacheEntry, Context = ExactCacheContext>> {
    service: Arc<ResponseCache<B>>,
    request: ResponseCacheRequest,
    key: String,
}

impl<B> ResponseCacheCall for NativeCacheCall<B>
where
    B: BaseCache<Value = CacheEntry, Context = ExactCacheContext>,
{
    fn lookup<'a>(&'a self, now: Duration) -> CacheFuture<'a, Option<CacheHit>> {
        Box::pin(async move {
            Ok(self
                .service
                .async_lookup_with_key(&self.request, || self.key.clone(), now)
                .await?
                .map(|value| CacheHit {
                    value,
                    key: self.key.clone(),
                }))
        })
    }

    fn store<'a>(&'a self, response: Value, now: Duration) -> CacheFuture<'a, ()> {
        Box::pin(self.service.async_store_with_key(
            &self.request,
            || self.key.clone(),
            response,
            now,
        ))
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CacheScope {
    Shared,
    Isolated(String),
}

#[derive(Clone, Copy, Default)]
pub struct CachePolicy {
    pub caching: Option<bool>,
    pub no_cache: bool,
    pub no_store: bool,
    pub ttl: Option<Duration>,
    pub max_age: Option<Duration>,
}

impl CachePolicy {
    pub fn reads(&self) -> bool {
        self.caching != Some(false) && !self.no_cache
    }

    pub fn writes(&self) -> bool {
        self.caching != Some(false) && !self.no_store
    }

    pub fn enabled(&self) -> bool {
        self.reads() || self.writes()
    }
}

#[derive(Clone)]
pub struct CacheOptions {
    pub policy: CachePolicy,
    pub scope: CacheScope,
}

impl CacheOptions {
    pub fn new(scope: CacheScope) -> Self {
        Self {
            policy: CachePolicy::default(),
            scope,
        }
    }

    pub fn request(self, namespace: &str, surface: &str, mut input: Value) -> ResponseCacheRequest {
        input.sort_all_objects();
        let scope = match self.scope {
            CacheScope::Shared => String::new(),
            CacheScope::Isolated(scope) => serde_json::json!(["isolated", scope]).to_string(),
        };
        ResponseCacheRequest {
            key: CacheKeyInput {
                namespace: Some(format!("{namespace}:inference-v2")),
                fields: [
                    ("surface", surface.to_owned()),
                    ("scope", scope),
                    ("request", input.to_string()),
                ]
                .into_iter()
                .map(|(name, value)| CacheKeyField {
                    name: name.into(),
                    value: Some(value),
                    api_parameter: true,
                    internal_parameter: false,
                })
                .collect(),
                ..Default::default()
            },
            controls: CacheControls {
                configured: true,
                supported_call_type: true,
                native_backend: true,
                default_on: true,
                caching: self.policy.caching,
                no_cache: self.policy.no_cache,
                no_store: self.policy.no_store,
                ..Default::default()
            },
            context: ExactCacheContext {
                ttl: self.policy.ttl,
            },
            max_age: self.policy.max_age,
        }
    }
}

#[derive(serde::Serialize, serde::Deserialize)]
pub struct ResponseEnvelope<T> {
    version: u32,
    surface: String,
    output: T,
}

impl<T> ResponseEnvelope<T> {
    pub fn new(surface: &str, output: T) -> Self {
        Self {
            version: 1,
            surface: surface.into(),
            output,
        }
    }

    pub fn decode(self, surface: &str) -> Option<T> {
        (self.version == 1 && self.surface == surface).then_some(self.output)
    }
}

#[derive(Clone)]
pub struct ScopedCache {
    pub service: std::sync::Arc<dyn ResponseCacheService>,
    pub scope: CacheScope,
}

impl ScopedCache {
    pub fn new(service: std::sync::Arc<dyn ResponseCacheService>, scope: CacheScope) -> Self {
        Self { service, scope }
    }

    pub fn options(&self, policy: Option<CachePolicy>) -> CacheOptions {
        CacheOptions {
            policy: policy.unwrap_or_default(),
            scope: self.scope.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;

    use litellm_cache_memory::InMemoryCache;
    use rstest::rstest;
    use serde_json::json;

    use super::*;

    #[rstest]
    #[case::shared(CacheScope::Shared)]
    #[case::isolated(CacheScope::Isolated("tenant".into()))]
    #[tokio::test]
    async fn native_preparation_keeps_existing_keys_and_storage(#[case] scope: CacheScope) {
        let service = Arc::new(
            ResponseCache::new(Arc::new(InMemoryCache::default())).with_config(
                ResponseCacheConfig {
                    namespace: "test".into(),
                    max_entry_bytes: 4096,
                },
            ),
        );
        let options = CacheOptions::new(scope);
        let input = json!({"model": "test", "body": {"z": 1, "a": 2}});
        let legacy = options.clone().request("test", "test", input.clone());
        let preparations = Cell::new(0);
        let prepared = service.clone().prepare(options, "test", &|| {
            preparations.set(preparations.get() + 1);
            input.clone()
        });
        assert_eq!(prepared.max_entry_bytes, 4096);
        assert_eq!(prepared.call.lookup(Duration::ZERO).await.unwrap(), None);
        prepared
            .call
            .store(json!({"answer": 42}), Duration::ZERO)
            .await
            .unwrap();
        assert_eq!(
            service.async_lookup(&legacy, Duration::ZERO).await.unwrap(),
            Some(json!({"answer": 42}))
        );
        assert_eq!(
            prepared.call.lookup(Duration::ZERO).await.unwrap(),
            Some(CacheHit {
                value: json!({"answer": 42}),
                key: cache_key(&legacy.key),
            })
        );
        assert_eq!(preparations.get(), 1);
    }

    #[rstest]
    #[tokio::test]
    async fn native_lookup_store_and_hit_reporting_hash_once() {
        let service = Arc::new(ResponseCache::new(Arc::new(InMemoryCache::default())));
        let before = crate::caching::KEY_DERIVATIONS.with(std::cell::Cell::get);
        let call = service
            .prepare(
                CacheOptions::new(CacheScope::Shared),
                "test",
                &|| json!({"prompt": "hello"}),
            )
            .call;

        assert_eq!(call.lookup(Duration::ZERO).await.unwrap(), None);
        call.store(json!({"answer": 42}), Duration::ZERO)
            .await
            .unwrap();
        let hit = call.lookup(Duration::ZERO).await.unwrap().unwrap();
        assert_eq!(hit.value, json!({"answer": 42}));
        assert!(!hit.key.is_empty());
        assert_eq!(
            crate::caching::KEY_DERIVATIONS.with(std::cell::Cell::get) - before,
            1
        );
    }

    #[rstest]
    #[tokio::test]
    async fn prepared_native_keys_keep_freshness_and_entry_limits() {
        let service = Arc::new(
            ResponseCache::new(Arc::new(InMemoryCache::default())).with_config(
                ResponseCacheConfig {
                    namespace: "test".into(),
                    max_entry_bytes: 16,
                },
            ),
        );
        let options = CacheOptions {
            policy: CachePolicy {
                max_age: Some(Duration::from_secs(1)),
                ..Default::default()
            },
            scope: CacheScope::Shared,
        };
        let input = json!({"body": "same request"});
        let request = options.clone().request("test", "test", input.clone());
        let call = service
            .clone()
            .prepare(options, "test", &|| input.clone())
            .call;
        call.store(json!({"answer": 42}), Duration::ZERO)
            .await
            .unwrap();
        let hit = call.lookup(Duration::from_secs(1)).await.unwrap().unwrap();

        assert_eq!(hit.key, cache_key(&request.key));
        assert_eq!(hit.value, json!({"answer": 42}));
        assert_eq!(call.lookup(Duration::from_secs(2)).await.unwrap(), None);

        call.store(
            json!({"answer": "too large for this cache"}),
            Duration::from_secs(2),
        )
        .await
        .unwrap();
        assert_eq!(call.lookup(Duration::from_secs(2)).await.unwrap(), None);
        assert_eq!(
            service
                .async_lookup(&request, Duration::ZERO)
                .await
                .unwrap(),
            Some(json!({"answer": 42}))
        );
    }
}
