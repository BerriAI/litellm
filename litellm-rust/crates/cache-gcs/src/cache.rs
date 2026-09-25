use std::{future::Future, sync::Arc, time::Duration};

use futures_util::future::try_join_all;
use litellm_cache::{
    BaseCache, BatchCache, BatchEntry, CacheCodec, DisconnectCache, Error, ExactCacheContext,
    FlushCache,
};
use percent_encoding::{AsciiSet, NON_ALPHANUMERIC, percent_encode};
use reqwest::Client;

use crate::{GcpTokenSource, TokenSource};

pub const DEFAULT_ENDPOINT: &str = "https://storage.googleapis.com";

const OBJECT_NAME_ENCODE_SET: &AsciiSet = &NON_ALPHANUMERIC
    .remove(b'-')
    .remove(b'_')
    .remove(b'.')
    .remove(b'~');

pub fn key_prefix(gcs_path: Option<&str>) -> String {
    match gcs_path {
        Some(path) if !path.is_empty() => format!("{}/", path.trim_end_matches('/')),
        _ => String::new(),
    }
}

#[derive(Clone, Debug)]
pub struct GcsConfig {
    pub bucket_name: String,
    pub gcs_path: Option<String>,
    pub path_service_account: Option<String>,
    pub endpoint: String,
}

impl GcsConfig {
    pub fn new(bucket_name: impl Into<String>) -> Self {
        Self {
            bucket_name: bucket_name.into(),
            gcs_path: None,
            path_service_account: None,
            endpoint: DEFAULT_ENDPOINT.to_string(),
        }
    }
}

pub struct GcsCache<S: CacheCodec> {
    config: GcsConfig,
    key_prefix: String,
    client: Client,
    token: Arc<dyn TokenSource>,
    codec: S,
}

impl<S: CacheCodec> GcsCache<S> {
    pub fn new(config: GcsConfig, client: Client, codec: S) -> Self {
        let token = Arc::new(GcpTokenSource::new(config.path_service_account.clone()));
        Self::with_token_source(config, client, codec, token)
    }

    pub fn with_token_source(
        config: GcsConfig,
        client: Client,
        codec: S,
        token: Arc<dyn TokenSource>,
    ) -> Self {
        let key_prefix = key_prefix(config.gcs_path.as_deref());
        Self {
            config,
            key_prefix,
            client,
            token,
            codec,
        }
    }

    pub fn bucket_name(&self) -> &str {
        &self.config.bucket_name
    }

    pub fn key_prefix(&self) -> &str {
        &self.key_prefix
    }

    pub fn path_service_account(&self) -> Option<&str> {
        self.config.path_service_account.as_deref()
    }

    pub fn object_name(&self, key: &str) -> String {
        format!("{}{}", self.key_prefix, key)
    }

    fn encoded_object_name(&self, key: &str) -> String {
        percent_encode(self.object_name(key).as_bytes(), OBJECT_NAME_ENCODE_SET).to_string()
    }

    fn endpoint(&self, path: &str) -> String {
        format!("{}{}", self.config.endpoint.trim_end_matches('/'), path)
    }

    async fn async_set(&self, key: &str, value: S::Value) -> Result<(), Error> {
        let token = self.token.bearer_token().await?;
        let payload = self.codec.encode(&value)?;
        let url = self.endpoint(&format!(
            "/upload/storage/v1/b/{}/o?uploadType=media&name={}",
            self.config.bucket_name,
            self.encoded_object_name(key)
        ));
        let response = self
            .client
            .post(url)
            .bearer_auth(token)
            .header(reqwest::header::CONTENT_TYPE, "application/json")
            .body(payload)
            .send()
            .await
            .map_err(|_| Error::Unavailable)?;
        if !response.status().is_success() {
            return Err(Error::Unavailable);
        }
        Ok(())
    }

    async fn async_get(&self, key: &str) -> Result<Option<S::Value>, Error> {
        let token = self.token.bearer_token().await?;
        let url = self.endpoint(&format!(
            "/storage/v1/b/{}/o/{}?alt=media",
            self.config.bucket_name,
            self.encoded_object_name(key)
        ));
        let response = self
            .client
            .get(url)
            .bearer_auth(token)
            .header(reqwest::header::CONTENT_TYPE, "application/json")
            .send()
            .await
            .map_err(|_| Error::Unavailable)?;
        if response.status() == reqwest::StatusCode::NOT_FOUND {
            return Ok(None);
        }
        if !response.status().is_success() {
            return Err(Error::Unavailable);
        }
        let body = response.bytes().await.map_err(|_| Error::Unavailable)?;
        self.codec
            .decode(&body)
            .map(Some)
            .map_err(|_| Error::InvalidEntry)
    }

    fn run_sync<T, F>(future: F) -> Result<T, Error>
    where
        F: Future<Output = Result<T, Error>> + Send,
        T: Send,
    {
        let run = |future: F| {
            tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .map_err(|_| Error::Unavailable)
                .and_then(|runtime| runtime.block_on(future))
        };
        match tokio::runtime::Handle::try_current() {
            Ok(handle) if handle.runtime_flavor() == tokio::runtime::RuntimeFlavor::MultiThread => {
                tokio::task::block_in_place(|| handle.block_on(future))
            }
            Ok(_) => std::thread::scope(|scope| {
                scope
                    .spawn(|| run(future))
                    .join()
                    .map_err(|_| Error::Unavailable)
                    .and_then(|result| result)
            }),
            Err(_) => run(future),
        }
    }
}

impl<S: CacheCodec> BaseCache for GcsCache<S> {
    type Value = S::Value;
    type Context = ExactCacheContext;

    fn get_ttl(&self, _: &Self::Context) -> Option<Duration> {
        None
    }

    fn set_cache(&self, key: &str, value: Self::Value, _: &Self::Context) -> Result<(), Error> {
        Self::run_sync(self.async_set(key, value))
    }

    fn get_cache(&self, key: &str, _: &Self::Context) -> Result<Option<Self::Value>, Error> {
        Self::run_sync(self.async_get(key))
    }

    async fn async_set_cache(
        &self,
        key: &str,
        value: Self::Value,
        _: Self::Context,
    ) -> Result<(), Error> {
        self.async_set(key, value).await
    }

    async fn async_get_cache(
        &self,
        key: &str,
        _: &Self::Context,
    ) -> Result<Option<Self::Value>, Error> {
        self.async_get(key).await
    }

    async fn async_set_cache_pipeline(
        &self,
        entries: Vec<(String, Self::Value)>,
        context: Self::Context,
    ) -> Result<(), Error> {
        try_join_all(entries.into_iter().map(|(key, value)| {
            let context = context.clone();
            async move { self.async_set_cache(&key, value, context).await }
        }))
        .await
        .map(|_| ())
    }
}

impl<S: CacheCodec> DisconnectCache for GcsCache<S> {
    async fn disconnect(&self) -> Result<(), Error> {
        Ok(())
    }
}

impl<S: CacheCodec> BatchCache for GcsCache<S> {
    async fn async_batch_get_cache(
        &self,
        keys: Vec<String>,
        context: Self::Context,
    ) -> Result<Vec<BatchEntry<Self::Value>>, Error> {
        try_join_all(keys.into_iter().map(|key| {
            let context = context.clone();
            async move {
                match self.async_get_cache(&key, &context).await {
                    Ok(Some(value)) => Ok(BatchEntry::Hit(value)),
                    Ok(None) => Ok(BatchEntry::Miss),
                    Err(Error::InvalidEntry) => Ok(BatchEntry::Invalid),
                    Err(error) => Err(error),
                }
            }
        }))
        .await
    }
}

impl<S: CacheCodec> FlushCache for GcsCache<S> {
    fn flush_cache(&self) -> Result<(), Error> {
        Ok(())
    }
}
