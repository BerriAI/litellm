use std::time::Duration;

use bytes::Bytes;

use litellm_http::Client;
use litellm_s3::S3Uri;
use litellm_s3::sdk::config::retry::RetryConfig;
use litellm_s3::sdk::config::timeout::TimeoutConfig;
use reqwest::header::{ETAG, HeaderMap, RETRY_AFTER};
use url::Url;

use crate::{Error, RETRYABLE_STATUSES, S3Access};

pub(crate) const MAX_BODY_BYTES: u64 = 100 * 1024 * 1024;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Fetched {
    pub body: Vec<u8>,
    pub etag: Option<String>,
}

#[derive(Debug)]
pub(crate) enum Cause {
    Status(u16),
    Transport(Error),
}

#[derive(Debug)]
pub(crate) struct Rejected {
    pub(crate) cause: Cause,
    pub(crate) retryable: bool,
    pub(crate) retry_after: Option<Duration>,
}

impl Rejected {
    pub(crate) fn fatal(error: Error) -> Self {
        Self {
            cause: Cause::Transport(error),
            retryable: false,
            retry_after: None,
        }
    }
}

/// Any status below 400 is the document
pub(crate) async fn http(
    client: &Client,
    url: Url,
    timeout: Duration,
) -> Result<Fetched, Rejected> {
    let transport = |error: litellm_http::Error| Rejected {
        cause: Cause::Transport(Error::from(error)),
        retryable: true,
        retry_after: None,
    };
    let mut response = tokio::time::timeout(timeout, client.get(url).send())
        .await
        .map_err(litellm_http::Error::read_timeout)
        .and_then(|sent| sent.map_err(litellm_http::Error::from))
        .map_err(transport)?;
    let status = response.status().as_u16();
    let etag = etag(response.headers());
    let retry_after = response
        .headers()
        .get(RETRY_AFTER)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.trim().parse().ok())
        .map(Duration::from_secs);
    if status >= 400 {
        return Err(Rejected {
            cause: Cause::Status(status),
            retryable: RETRYABLE_STATUSES.contains(&status),
            retry_after,
        });
    }
    if response
        .content_length()
        .is_some_and(|length| length > MAX_BODY_BYTES)
    {
        return Err(Rejected::fatal(Error::body_too_large(MAX_BODY_BYTES)));
    }
    let mut body = Vec::new();
    while let Some(chunk) = tokio::time::timeout(timeout, response.chunk())
        .await
        .map_err(litellm_http::Error::body_read_timeout)
        .and_then(|read| read.map_err(litellm_http::Error::from))
        .map_err(transport)?
    {
        append(&mut body, &chunk).map_err(Rejected::fatal)?;
    }
    Ok(Fetched { body, etag })
}

/// S3 failures are final: the SDK already classifies its own errors and a missing or denied object stays so
pub(crate) async fn s3(
    client: &Client,
    s3: &S3Access,
    object: &S3Uri,
    timeout: Duration,
) -> Result<Fetched, Rejected> {
    let fetched = async {
        let config = match s3 {
            S3Access::Environment { region } => {
                litellm_s3::environment_config(client.clone(), region.clone()).await
            }
            S3Access::Config(config) => config.to_builder(),
        }
        .retry_config(RetryConfig::disabled())
        .timeout_config(
            TimeoutConfig::builder()
                .operation_attempt_timeout(timeout)
                .build(),
        )
        .build();
        let output = litellm_s3::sdk::Client::from_conf(config)
            .get_object()
            .bucket(&object.bucket)
            .key(&object.key)
            .send()
            .await?;
        if output
            .content_length()
            .is_some_and(|length| length.unsigned_abs() > MAX_BODY_BYTES)
        {
            return Err(Error::body_too_large(MAX_BODY_BYTES));
        }
        let etag = output.e_tag().map(str::to_owned);
        let mut stream = output.body;
        let mut body = Vec::new();
        while let Some(chunk) = tokio::time::timeout(timeout, stream.try_next()).await?? {
            append(&mut body, &chunk)?;
        }
        Ok::<_, Error>(Fetched { body, etag })
    };
    fetched.await.map_err(Rejected::fatal)
}

fn append(body: &mut Vec<u8>, chunk: &Bytes) -> Result<(), Error> {
    if (body.len() + chunk.len()) as u64 > MAX_BODY_BYTES {
        return Err(Error::body_too_large(MAX_BODY_BYTES));
    }
    body.extend_from_slice(chunk);
    Ok(())
}

fn etag(headers: &HeaderMap) -> Option<String> {
    let values: Vec<&str> = headers
        .get_all(ETAG)
        .iter()
        .filter_map(|value| value.to_str().ok())
        .collect();
    (!values.is_empty()).then(|| values.join(", "))
}
