use std::{future::Future, pin::Pin, time::Duration};

use litellm_cache::{BaseCache, Error, ExactCacheContext};
use serde_json::Value;

use crate::{
    CacheControls, CacheEntry, CacheKeyField, CacheKeyInput, ResponseCache, ResponseCacheRequest,
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

pub trait ResponseCacheService: Send + Sync {
    fn config(&self) -> &ResponseCacheConfig;

    fn lookup<'a>(
        &'a self,
        request: &'a ResponseCacheRequest,
        now: Duration,
    ) -> CacheFuture<'a, Option<Value>>;

    fn store<'a>(
        &'a self,
        request: &'a ResponseCacheRequest,
        response: Value,
        now: Duration,
    ) -> CacheFuture<'a, ()>;
}

impl<B> ResponseCacheService for ResponseCache<B>
where
    B: BaseCache<Value = CacheEntry, Context = ExactCacheContext>,
{
    fn config(&self) -> &ResponseCacheConfig {
        self.config()
    }

    fn lookup<'a>(
        &'a self,
        request: &'a ResponseCacheRequest,
        now: Duration,
    ) -> CacheFuture<'a, Option<Value>> {
        Box::pin(self.async_lookup(request, now))
    }

    fn store<'a>(
        &'a self,
        request: &'a ResponseCacheRequest,
        response: Value,
        now: Duration,
    ) -> CacheFuture<'a, ()> {
        Box::pin(self.async_store(request, response, now))
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CacheScope {
    Shared,
    Isolated(String),
}

#[derive(Clone)]
pub struct CacheOptions {
    pub caching: Option<bool>,
    pub no_cache: bool,
    pub no_store: bool,
    pub ttl: Option<Duration>,
    pub max_age: Option<Duration>,
    pub scope: CacheScope,
}

impl CacheOptions {
    pub fn new(scope: CacheScope) -> Self {
        Self {
            caching: None,
            no_cache: false,
            no_store: false,
            ttl: None,
            max_age: None,
            scope,
        }
    }

    pub fn enabled(&self) -> bool {
        self.caching != Some(false) && !(self.no_cache && self.no_store)
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
                caching: self.caching,
                no_cache: self.no_cache,
                no_store: self.no_store,
                ..Default::default()
            },
            context: ExactCacheContext { ttl: self.ttl },
            max_age: self.max_age,
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

    pub fn options(&self, overrides: Option<CacheOptions>) -> CacheOptions {
        overrides.unwrap_or_else(|| CacheOptions::new(self.scope.clone()))
    }
}
