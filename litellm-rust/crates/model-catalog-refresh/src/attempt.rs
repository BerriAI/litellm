use std::time::Duration;

use litellm_http::Client;
use litellm_s3::ProvideCredentials;
use litellm_s3::S3Uri;
use litellm_s3::sdk::config::retry::RetryConfig;
use litellm_s3::sdk::config::timeout::TimeoutConfig;
use reqwest::header::{ETAG, HeaderMap, RETRY_AFTER};
use url::Url;

use crate::{Error, RETRYABLE_STATUSES, Target};

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

/// Any status below 400 is the document, redirects included, since the caller's client does not follow them
pub(crate) async fn http(
    client: &Client,
    url: Url,
    timeout: Duration,
) -> Result<Fetched, Rejected> {
    let transport = |error: reqwest::Error| Rejected {
        cause: Cause::Transport(Error::from(litellm_http::Error::from(error))),
        retryable: true,
        retry_after: None,
    };
    let response = client
        .get(url)
        .timeout(timeout)
        .send()
        .await
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
    let body = response.bytes().await.map_err(transport)?;
    Ok(Fetched {
        body: body.to_vec(),
        etag,
    })
}

/// S3 failures are final: the SDK already classifies its own errors and a missing or denied object stays so
pub(crate) async fn s3(
    client: &Client,
    credentials: &(impl ProvideCredentials + Clone + 'static),
    target: &Target,
    object: &S3Uri,
    timeout: Duration,
) -> Result<Fetched, Rejected> {
    let config = litellm_s3::config(
        target.aws_region.clone(),
        target.aws_endpoint_url.as_deref(),
        client.clone(),
        credentials.clone(),
    )
    .retry_config(RetryConfig::disabled())
    .timeout_config(
        TimeoutConfig::builder()
            .operation_attempt_timeout(timeout)
            .build(),
    )
    .build();
    let fetched = async {
        let output = litellm_s3::sdk::Client::from_conf(config)
            .get_object()
            .bucket(&object.bucket)
            .key(&object.key)
            .send()
            .await?;
        let etag = output.e_tag().map(str::to_owned);
        let body = output.body.collect().await?;
        Ok::<_, Error>(Fetched {
            body: body.to_vec(),
            etag,
        })
    };
    tokio::time::timeout(timeout, fetched)
        .await
        .unwrap_or_else(|elapsed| Err(Error::from(elapsed)))
        .map_err(Rejected::fatal)
}

fn etag(headers: &HeaderMap) -> Option<String> {
    let values: Vec<&str> = headers
        .get_all(ETAG)
        .iter()
        .filter_map(|value| value.to_str().ok())
        .collect();
    (!values.is_empty()).then(|| values.join(", "))
}
