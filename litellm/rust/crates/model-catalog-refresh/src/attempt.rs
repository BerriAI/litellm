use std::time::Duration;

use bytes::Bytes;

use litellm_auth_gcp::{CredentialSource, GcpAuth};
use litellm_http::Client;
use litellm_s3::S3Uri;
use litellm_s3::sdk::config::retry::RetryConfig;
use litellm_s3::sdk::config::timeout::TimeoutConfig;
use percent_encoding::{AsciiSet, NON_ALPHANUMERIC, utf8_percent_encode};
use reqwest::header::{ETAG, HeaderMap, RETRY_AFTER};
use url::Url;

use crate::{Error, GcsAccess, GcsUri, RETRYABLE_STATUSES, S3Access};

const GCS_ENDPOINT: &str = "https://storage.googleapis.com";

const OBJECT_NAME_ENCODE_SET: &AsciiSet = &NON_ALPHANUMERIC
    .remove(b'-')
    .remove(b'_')
    .remove(b'.')
    .remove(b'~');

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
    bearer: Option<&str>,
    timeout: Duration,
) -> Result<Fetched, Rejected> {
    let request = match bearer {
        Some(token) => client.get(url).bearer_auth(token),
        None => client.get(url),
    };
    let transport = |error: litellm_http::Error| Rejected {
        cause: Cause::Transport(Error::from(error)),
        retryable: true,
        retry_after: None,
    };
    let mut response = tokio::time::timeout(timeout, request.send())
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

pub(crate) async fn gcs(
    client: &Client,
    gcs: &GcsAccess,
    object: &GcsUri,
    timeout: Duration,
) -> Result<Fetched, Rejected> {
    let started = tokio::time::Instant::now();
    let (endpoint, token) = match gcs {
        GcsAccess::Environment => {
            match emulator_endpoint(std::env::var("STORAGE_EMULATOR_HOST").ok()) {
                Some(emulator) => (emulator, None),
                None => (
                    GCS_ENDPOINT.to_owned(),
                    Some(
                        bounded_token(
                            timeout,
                            GcpAuth::default().access_token(CredentialSource::application_default(
                                &|name| std::env::var(name).ok(),
                            )),
                        )
                        .await?,
                    ),
                ),
            }
        }
        GcsAccess::Config {
            endpoint,
            token_source,
        } => (
            endpoint.clone(),
            Some(bounded_token(timeout, token_source.token()).await?),
        ),
    };
    let url = Url::parse(&format!(
        "{}/storage/v1/b/{}/o/{}?alt=media",
        endpoint.trim_end_matches('/'),
        object.bucket,
        utf8_percent_encode(&object.object, OBJECT_NAME_ENCODE_SET),
    ))
    .map_err(|error| Rejected::fatal(Error::from(error)))?;
    http(
        client,
        url,
        token.as_ref().map(|token| token.expose()),
        timeout.saturating_sub(started.elapsed()),
    )
    .await
}

async fn bounded_token<T>(
    timeout: Duration,
    token: impl Future<Output = Result<T, litellm_auth_types::Error>>,
) -> Result<T, Rejected> {
    let retryable = |error| Rejected {
        cause: Cause::Transport(error),
        retryable: true,
        retry_after: None,
    };
    tokio::time::timeout(timeout, token)
        .await
        .map_err(|elapsed| retryable(Error::gcs_token_timeout(elapsed)))?
        .map_err(|error| retryable(Error::gcs_token(error)))
}

fn emulator_endpoint(host: Option<String>) -> Option<String> {
    match host {
        Some(host) if host.contains("://") => Some(host),
        Some(host) if !host.is_empty() => Some(format!("http://{host}")),
        _ => None,
    }
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
