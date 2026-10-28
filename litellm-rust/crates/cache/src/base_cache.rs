use std::{future::Future, time::Duration};

use serde::{Deserialize, Serialize};

use crate::Error;

#[derive(Clone, Debug, PartialEq)]
pub enum BatchEntry<V> {
    Hit(V),
    Miss,
    Invalid,
}

pub trait CacheContext: Clone + Send + Sync + 'static {
    fn ttl(&self) -> Option<Duration>;

    fn with_ttl(&self, ttl: Option<Duration>) -> Self;
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ExactCacheContext {
    pub ttl: Option<Duration>,
}

impl CacheContext for ExactCacheContext {
    fn ttl(&self) -> Option<Duration> {
        self.ttl
    }

    fn with_ttl(&self, ttl: Option<Duration>) -> Self {
        Self { ttl }
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct SemanticCacheContext {
    pub input: Option<serde_json::Value>,
    pub messages: Option<serde_json::Value>,
    pub metadata: Option<serde_json::Value>,
    pub scope: Option<String>,
    pub ttl: Option<Duration>,
}

impl CacheContext for SemanticCacheContext {
    fn ttl(&self) -> Option<Duration> {
        self.ttl
    }

    fn with_ttl(&self, ttl: Option<Duration>) -> Self {
        Self {
            ttl,
            ..self.clone()
        }
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum CacheConnectionStatus {
    Success,
    Failed,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct CacheConnectionResult {
    pub status: CacheConnectionStatus,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

pub trait BaseCache: Send + Sync {
    type Value: Clone + Send + Sync + 'static;
    type Context: CacheContext;

    fn get_ttl(&self, context: &Self::Context) -> Option<Duration>;

    fn set_cache(
        &self,
        key: &str,
        value: Self::Value,
        context: &Self::Context,
    ) -> Result<(), Error>;

    fn get_cache(&self, key: &str, context: &Self::Context) -> Result<Option<Self::Value>, Error>;

    fn async_set_cache(
        &self,
        key: &str,
        value: Self::Value,
        context: Self::Context,
    ) -> impl Future<Output = Result<(), Error>> + Send {
        async move { self.set_cache(key, value, &context) }
    }

    fn async_get_cache(
        &self,
        key: &str,
        context: &Self::Context,
    ) -> impl Future<Output = Result<Option<Self::Value>, Error>> + Send {
        async move { self.get_cache(key, context) }
    }

    fn async_set_cache_pipeline(
        &self,
        entries: Vec<(String, Self::Value)>,
        context: Self::Context,
    ) -> impl Future<Output = Result<(), Error>> + Send {
        async move {
            for (key, value) in entries {
                self.async_set_cache(&key, value, context.clone()).await?;
            }
            Ok(())
        }
    }

    fn batch_cache_write(
        &self,
        key: &str,
        value: Self::Value,
        context: Self::Context,
    ) -> impl Future<Output = Result<(), Error>> + Send {
        self.async_set_cache(key, value, context)
    }
}
