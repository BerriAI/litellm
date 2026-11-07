use std::time::Duration;

use litellm_error::Report;
use litellm_http::Client;
use litellm_s3::ProvideCredentials;
use litellm_s3::sdk::config::retry::RetryConfig;
use litellm_s3::sdk::config::timeout::TimeoutConfig;
use reqwest::header::{ETAG, HeaderMap, RETRY_AFTER};
use url::Url;

use litellm_s3::S3Uri;

use crate::{Attempts, Cause, Error, Failure, Fetched, Reply, RetryPolicy, Source, Step};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Target {
    pub url: String,
    pub aws_region: String,
    pub aws_endpoint_url: Option<String>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Failed {
    pub reason: String,
    /// Another attempt could succeed. Only the attempt limit ended the fetch
    pub retryable: bool,
    pub retry_after: Option<f64>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Start {
    Fresh,
    /// Continues a fetch whose first attempt failed retryably elsewhere, waiting for that failure first
    After(Failed),
}

pub async fn fetch(
    client: &Client,
    credentials: &(impl ProvideCredentials + Clone + 'static),
    target: &Target,
    policy: RetryPolicy,
    start: Start,
) -> Result<Fetched, Failed> {
    let attempts = match start {
        Start::Fresh => Attempts::new(policy),
        Start::After(failed) => {
            let Some((attempts, wait)) =
                Attempts::resume(policy, failed.retry_after, rand::random())
            else {
                return Err(failed);
            };
            tokio::time::sleep(wait).await;
            attempts
        }
    };
    let source = match Source::parse(&target.url) {
        Ok(source) => source,
        Err(error) => {
            return Err(Failed::of(
                target,
                false,
                &Failure {
                    cause: Cause::Transport(error),
                    attempts: 1,
                    retryable: false,
                    retry_after: None,
                },
            ));
        }
    };
    let s3 = matches!(source, Source::S3(_));
    retry(client, credentials, target, &source, attempts)
        .await
        .map_err(|failure| Failed::of(target, s3, &failure))
}

async fn retry(
    client: &Client,
    credentials: &(impl ProvideCredentials + Clone + 'static),
    target: &Target,
    source: &Source,
    mut attempts: Attempts,
) -> Result<Fetched, Failure> {
    loop {
        let reply = match source {
            Source::Http(url) => http_attempt(client, url.clone(), attempts.timeout()).await,
            Source::S3(object) => {
                s3_attempt(client, credentials, target, object, attempts.timeout()).await
            }
        };
        let reason = describe(&reply, &target.url);
        let attempt = attempts.made() + 1;
        match attempts.next(reply, rand::random()) {
            Step::Sleep(wait) => {
                litellm_tracing::warn!(
                    "LiteLLM: model cost map fetch attempt {attempt}/{} failed ({reason}); retrying in {:.1}s",
                    attempts.max_attempts(),
                    wait.as_secs_f64(),
                );
                tokio::time::sleep(wait).await;
            }
            Step::Done(result) => return result,
        }
    }
}

async fn http_attempt(client: &Client, url: Url, timeout: Duration) -> Reply {
    let response = match client.get(url).timeout(timeout).send().await {
        Ok(response) => response,
        Err(error) => return transport(error, true),
    };
    let status = response.status().as_u16();
    let etag = etag(response.headers());
    let retry_after = response
        .headers()
        .get(RETRY_AFTER)
        .and_then(|value| value.to_str().ok())
        .and_then(litellm_python_compat::number::float)
        .filter(|seconds| *seconds >= 0.0);
    match response.bytes().await {
        Ok(body) => Reply::Response {
            status,
            etag,
            retry_after,
            body: body.to_vec(),
        },
        Err(error) => transport(error, true),
    }
}

/// One `GetObject`. Every failure is final, as boto3's were on main
async fn s3_attempt(
    client: &Client,
    credentials: &(impl ProvideCredentials + Clone + 'static),
    target: &Target,
    object: &S3Uri,
    timeout: Duration,
) -> Reply {
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
        Ok::<_, Error>((etag, body.to_vec()))
    };
    match tokio::time::timeout(timeout, fetched)
        .await
        .unwrap_or_else(|elapsed| Err(Error::from(elapsed)))
    {
        Ok((etag, body)) => Reply::Response {
            status: 200,
            etag,
            retry_after: None,
            body,
        },
        Err(error) => Reply::Transport {
            error,
            retryable: false,
        },
    }
}

/// Every `ETag` value joined by `", "`, as main read them
fn etag(headers: &HeaderMap) -> Option<String> {
    let values: Vec<&str> = headers
        .get_all(ETAG)
        .iter()
        .filter_map(|value| value.to_str().ok())
        .collect();
    (!values.is_empty()).then(|| values.join(", "))
}

fn transport(error: reqwest::Error, retryable: bool) -> Reply {
    Reply::Transport {
        error: Error::from(litellm_http::Error::from(error)),
        retryable,
    }
}

fn describe(reply: &Reply, url: &str) -> String {
    match reply {
        Reply::Response { status, .. } => format!("HTTP {status} from {url}"),
        Reply::Transport { error, .. } => format!("error fetching {url}: {}", Report::new(error)),
    }
}

impl Failed {
    fn of(target: &Target, s3: bool, failure: &Failure) -> Self {
        let url = &target.url;
        let cause = match &failure.cause {
            Cause::Status(status) => format!("HTTP {status} from {url}"),
            Cause::Transport(error) if s3 => {
                format!("S3 error fetching {url}: {}", Report::new(error))
            }
            Cause::Transport(error) => format!("error fetching {url}: {}", Report::new(error)),
        };
        let reason = if failure.retryable && failure.attempts > 1 {
            format!("{cause} (after {} attempts)", failure.attempts)
        } else {
            cause
        };
        Self {
            reason,
            retryable: failure.retryable,
            retry_after: failure.retry_after,
        }
    }
}
