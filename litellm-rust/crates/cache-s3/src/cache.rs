use std::{
    future::Future,
    sync::Arc,
    time::{Duration, SystemTime},
};

use aws_sdk_s3::{
    config::{BehaviorVersion, Region, RequestChecksumCalculation, ResponseChecksumValidation},
    error::SdkError,
    primitives::ByteStream,
};
use aws_smithy_types::{DateTime, date_time::Format};
use futures_util::future::try_join_all;
use litellm_auth_aws::AwsAuthConfig;
use litellm_cache::{
    BaseCache, BatchCache, CacheCodec, DisconnectCache, Error, ExactCacheContext, FlushCache,
};
use tokio::runtime::Handle;

use crate::{auth::S3Credentials, transport::ReqwestHttpClient};

pub struct S3Endpoint {
    pub url: String,
}

pub struct S3CacheConfig {
    pub bucket: String,
    pub key_prefix: String,
    pub region: String,
    pub endpoint: Option<S3Endpoint>,
    pub auth: AwsAuthConfig,
}

pub struct S3Cache<C: CacheCodec> {
    client: aws_sdk_s3::Client,
    codec: C,
    runtime: Handle,
    bucket: Arc<str>,
    key_prefix: Arc<str>,
    region: Arc<str>,
    endpoint: Option<Arc<str>>,
}

impl<C: CacheCodec> S3Cache<C> {
    pub fn new(config: S3CacheConfig, http: reqwest::Client, codec: C, runtime: Handle) -> Self {
        let endpoint_url: Option<String> = config.endpoint.map(|endpoint| endpoint.url);
        let base = aws_sdk_s3::Config::builder()
            .behavior_version(BehaviorVersion::latest())
            .region(Region::new(config.region.clone()))
            .http_client(ReqwestHttpClient(http))
            .credentials_provider(S3Credentials::new(config.auth))
            .request_checksum_calculation(RequestChecksumCalculation::WhenRequired)
            .response_checksum_validation(ResponseChecksumValidation::WhenRequired);
        let builder = match &endpoint_url {
            Some(url) => base.endpoint_url(url).force_path_style(true),
            None => base,
        };
        Self {
            client: aws_sdk_s3::Client::from_conf(builder.build()),
            codec,
            runtime,
            bucket: config.bucket.into(),
            key_prefix: config.key_prefix.into(),
            region: config.region.into(),
            endpoint: endpoint_url.map(Into::into),
        }
    }

    pub fn bucket(&self) -> &str {
        &self.bucket
    }

    pub fn key_prefix(&self) -> &str {
        &self.key_prefix
    }

    pub fn region(&self) -> &str {
        &self.region
    }

    pub fn endpoint(&self) -> Option<&str> {
        self.endpoint.as_deref()
    }

    pub fn to_s3_key(&self, key: &str) -> String {
        format!("{}{}", self.key_prefix, key.replace(':', "/"))
    }

    fn block_on<F: Future>(&self, future: F) -> F::Output {
        if Handle::try_current().is_ok() {
            tokio::task::block_in_place(|| self.runtime.block_on(future))
        } else {
            self.runtime.block_on(future)
        }
    }

    async fn put(
        &self,
        key: &str,
        value: C::Value,
        context: &ExactCacheContext,
    ) -> Result<(), Error> {
        let s3_key = self.to_s3_key(key);
        let body = self.codec.encode(&value)?;
        let request = self
            .client
            .put_object()
            .bucket(self.bucket.as_ref())
            .key(&s3_key)
            .body(ByteStream::from(body))
            .content_type("application/json")
            .content_language("en")
            .content_disposition(format!("inline; filename=\"{s3_key}.json\""));
        let request = match context.ttl {
            Some(ttl) => {
                let seconds = ttl.as_secs_f64();
                request
                    .cache_control(format!("immutable, max-age={seconds}, s-maxage={seconds}"))
                    .expires(DateTime::from(SystemTime::now() + ttl))
            }
            None => request.cache_control("immutable, max-age=31536000, s-maxage=31536000"),
        };
        request.send().await.map_err(|_| Error::Unavailable)?;
        Ok(())
    }

    async fn get(&self, key: &str) -> Result<Option<C::Value>, Error> {
        let output = match self
            .client
            .get_object()
            .bucket(self.bucket.as_ref())
            .key(self.to_s3_key(key))
            .send()
            .await
        {
            Ok(output) => output,
            Err(error) => {
                if let SdkError::ServiceError(service) = &error {
                    let status = error
                        .raw_response()
                        .map(|response| response.status().as_u16());
                    let not_found = service.err().is_no_such_key()
                        || service.err().meta().code() == Some("AccessDenied")
                        || status == Some(404)
                        || status == Some(403);
                    if not_found {
                        return Ok(None);
                    }
                }
                return Err(Error::Unavailable);
            }
        };
        if let Some(expires) = output.expires_string()
            && let Ok(expires) = DateTime::from_str(expires, Format::HttpDate)
            && expires < DateTime::from(SystemTime::now())
        {
            return Ok(None);
        }
        let bytes = output
            .body
            .collect()
            .await
            .map_err(|_| Error::Unavailable)?
            .into_bytes();
        self.codec.decode(&bytes).map(Some)
    }
}

impl<C: CacheCodec> BaseCache for S3Cache<C> {
    type Value = C::Value;
    type Context = ExactCacheContext;

    fn get_ttl(&self, context: &Self::Context) -> Option<Duration> {
        context.ttl
    }

    fn set_cache(
        &self,
        key: &str,
        value: Self::Value,
        context: &Self::Context,
    ) -> Result<(), Error> {
        self.block_on(self.put(key, value, context))
    }

    fn get_cache(&self, key: &str, _context: &Self::Context) -> Result<Option<Self::Value>, Error> {
        self.block_on(self.get(key))
    }

    async fn async_set_cache(
        &self,
        key: &str,
        value: Self::Value,
        context: Self::Context,
    ) -> Result<(), Error> {
        self.put(key, value, &context).await
    }

    async fn async_get_cache(
        &self,
        key: &str,
        _context: &Self::Context,
    ) -> Result<Option<Self::Value>, Error> {
        self.get(key).await
    }

    async fn async_set_cache_pipeline(
        &self,
        entries: Vec<(String, Self::Value)>,
        context: Self::Context,
    ) -> Result<(), Error> {
        let context = &context;
        try_join_all(
            entries
                .into_iter()
                .map(|(key, value)| async move { self.put(&key, value, context).await }),
        )
        .await
        .map(drop)
    }
}

impl<C: CacheCodec> DisconnectCache for S3Cache<C> {
    async fn disconnect(&self) -> Result<(), Error> {
        Ok(())
    }
}

impl<C: CacheCodec> BatchCache for S3Cache<C> {}

impl<C: CacheCodec> FlushCache for S3Cache<C> {
    fn flush_cache(&self) -> Result<(), Error> {
        Ok(())
    }
}
