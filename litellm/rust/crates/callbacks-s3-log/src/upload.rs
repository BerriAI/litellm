use std::{
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
    time::{Duration, SystemTime},
};

use base64::Engine;
use bytes::Bytes;
use futures_util::FutureExt;
use litellm_auth_aws::{AwsAuthService, Credentials};
use litellm_callbacks::{PayloadRef, RenderOptions, render_all};
use litellm_s3::signed::{Reply, SendError, SignedRequest, send};
use reqwest::Method;

use crate::{
    key::{hex_sha256, object_url},
    limiter::Limiter,
    settings::{Environment, Settings},
    sink::Host,
};

const ATTEMPTS: u32 = 3;
const RETRYABLE: [u16; 4] = [403, 429, 500, 503];
const TERMINAL_CODES: [&str; 7] = [
    "EntityTooLarge",
    "InvalidArgument",
    "MalformedXML",
    "InvalidDigest",
    "KeyTooLongError",
    "BadDigest",
    "InvalidRequest",
];
const CACHE_CONTROL: &str = "private, immutable, max-age=31536000, s-maxage=0";
const EMPTY_SHA256: &str = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Outcome {
    Delivered,
    Retry,
    Dropped,
}

pub struct Held {
    payload: Arc<PayloadRef>,
    options: RenderOptions,
    live: Arc<AtomicUsize>,
}

impl Held {
    pub fn new(payload: Arc<PayloadRef>, options: RenderOptions, live: Arc<AtomicUsize>) -> Self {
        live.fetch_add(1, Ordering::Relaxed);
        Self {
            payload,
            options,
            live,
        }
    }
}

impl std::fmt::Debug for Held {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("Held")
            .field("payload", &self.payload)
            .field("options", &self.options)
            .finish()
    }
}

impl Drop for Held {
    fn drop(&mut self) {
        self.live.fetch_sub(1, Ordering::Relaxed);
    }
}

#[derive(Clone, Debug)]
pub enum Body {
    Ready(Bytes),
    Payload(Arc<Held>),
}

#[derive(Clone, Debug)]
pub struct Object {
    pub key: String,
    pub body: Body,
    pub content_type: &'static str,
    pub download_name: String,
}

pub async fn on_host<T: Send + 'static>(
    host: &Arc<dyn Host>,
    call: impl FnOnce(&dyn Host) -> T + Send + 'static,
) -> Result<T, String> {
    let host = Arc::clone(host);
    tokio::task::spawn_blocking(move || call(host.as_ref()))
        .await
        .map_err(|error| error.to_string())
}

pub fn panic_message(panic: &(dyn std::any::Any + Send)) -> String {
    panic
        .downcast_ref::<&str>()
        .map(|message| (*message).to_owned())
        .or_else(|| panic.downcast_ref::<String>().cloned())
        .unwrap_or_else(|| "a panic without a message".to_owned())
}

pub async fn render_bodies(bodies: &[&Body]) -> Result<Vec<Bytes>, String> {
    let held: Vec<&Held> = bodies
        .iter()
        .filter_map(|body| match body {
            Body::Payload(held) => Some(held.as_ref()),
            Body::Ready(_) => None,
        })
        .collect();
    let mut groups: Vec<(&RenderOptions, Vec<usize>)> = Vec::new();
    for (index, item) in held.iter().enumerate() {
        match groups
            .iter_mut()
            .find(|(options, _)| **options == item.options)
        {
            Some((_, indexes)) => indexes.push(index),
            None => groups.push((&item.options, vec![index])),
        }
    }
    let mut rendered: Vec<Option<Bytes>> = vec![None; held.len()];
    for (options, indexes) in groups {
        let payloads: Vec<&PayloadRef> = indexes
            .iter()
            .map(|index| held[*index].payload.as_ref())
            .collect();
        let encoded = std::panic::AssertUnwindSafe(render_all(&payloads, options.clone()))
            .catch_unwind()
            .await
            .map_err(|panic| panic_message(panic.as_ref()))??;
        for (index, bytes) in indexes.into_iter().zip(encoded) {
            rendered[index] = Some(Bytes::from(bytes));
        }
    }
    let mut rendered = rendered.into_iter();
    bodies
        .iter()
        .map(|body| match body {
            Body::Ready(bytes) => Ok(bytes.clone()),
            Body::Payload(_) => rendered
                .next()
                .flatten()
                .ok_or_else(|| "the host rendered too few payloads".to_owned()),
        })
        .collect()
}

impl Body {
    pub async fn render(&self) -> Result<Bytes, String> {
        render_bodies(&[self])
            .await?
            .pop()
            .ok_or_else(|| "the host rendered no payload".to_owned())
    }
}

pub type Clock = fn() -> SystemTime;
pub type Backoff = Arc<dyn Fn(u32) -> Duration + Send + Sync>;

pub struct Uploader {
    pub settings: Arc<Settings>,
    pub client: litellm_http::Client,
    pub download_client: litellm_http::Client,
    pub host: Arc<dyn Host>,
    pub auth: AwsAuthService,
    pub environment: Arc<Environment>,
    pub clock: Clock,
    pub backoff: Backoff,
    pub retries: AtomicUsize,
}

enum Attempt {
    Reply(Reply),
    Failed(SendError),
}

pub fn error_code(body: &str) -> Option<&str> {
    let mut rest = body;
    while let Some(start) = rest.find("<Code>") {
        let after = &rest[start + "<Code>".len()..];
        let code_length = after.find('<').unwrap_or(after.len());
        if code_length > 0 && after[code_length..].starts_with("</Code>") {
            return Some(&after[..code_length]);
        }
        rest = after;
    }
    None
}

pub fn is_terminal(status: u16, body: &str) -> bool {
    matches!(status, 400 | 403)
        && error_code(body).is_some_and(|code| TERMINAL_CODES.contains(&code))
}

pub fn python_backoff(attempt: u32) -> Duration {
    Duration::from_secs_f64(f64::from(1u32 << attempt) * (0.5 + rand::random::<f64>()))
}

fn throttled(attempt: &Attempt) -> bool {
    match attempt {
        Attempt::Failed(_) => true,
        Attempt::Reply(reply) => {
            matches!(reply.status, 429 | 503) || error_code(&reply.text()) == Some("SlowDown")
        }
    }
}

impl Uploader {
    pub fn headers(&self, object: &Object, body: &Bytes) -> Vec<(String, String)> {
        let md5 = base64::engine::general_purpose::STANDARD
            .encode(<md5::Md5 as md5::Digest>::digest(body));
        let mut headers = vec![
            ("Content-Type".to_owned(), object.content_type.to_owned()),
            ("Content-MD5".to_owned(), md5),
            ("x-amz-content-sha256".to_owned(), hex_sha256(body)),
            ("Content-Language".to_owned(), "en".to_owned()),
            (
                "Content-Disposition".to_owned(),
                format!("inline; filename=\"{}\"", object.download_name),
            ),
            ("Cache-Control".to_owned(), CACHE_CONTROL.to_owned()),
        ];
        if let Some(algorithm) = &self.settings.server_side_encryption {
            headers.push(("x-amz-server-side-encryption".to_owned(), algorithm.clone()));
        }
        if let Some(key_id) = &self.settings.sse_kms_key_id {
            headers.push((
                "x-amz-server-side-encryption-aws-kms-key-id".to_owned(),
                key_id.clone(),
            ));
        }
        headers
    }

    async fn credentials(&self) -> Result<Credentials, String> {
        let environment = Arc::clone(&self.environment);
        let lookup = move |name: &str| environment(name);
        self.auth
            .resolve_credentials(self.settings.auth.clone(), &lookup)
            .await
            .map_err(|error| error.to_string())
    }

    async fn forget_credentials(&self) {
        let environment = Arc::clone(&self.environment);
        let lookup = move |name: &str| environment(name);
        self.auth
            .invalidate_credentials(self.settings.auth.clone(), &lookup)
            .await;
    }

    async fn attempt(
        &self,
        url: &str,
        headers: &[(String, String)],
        body: &Bytes,
    ) -> Result<Attempt, String> {
        let credentials = self.credentials().await?;
        let from_environment = match self.settings.region_name {
            Some(_) => Ok(None),
            None => on_host(&self.host, |host| host.region_from_environment())
                .await
                .and_then(|region| region),
        };
        let region = self.settings.signing_region(|| from_environment)?;
        let put = || {
            send(
                &self.client,
                SignedRequest {
                    method: Method::PUT,
                    url,
                    headers,
                    body: body.clone(),
                    region: &region,
                    credentials: &credentials,
                    time: (self.clock)(),
                    read_timeout: Some(self.settings.attempt_timeout),
                },
            )
        };
        let sent = match put().await {
            Err(error) if error.is_reconnectable() => put().await,
            sent => sent,
        };
        match sent {
            Ok(reply) => Ok(Attempt::Reply(reply)),
            Err(error) if error.is_transient() => Ok(Attempt::Failed(error)),
            Err(error) => Err(error.to_string()),
        }
    }

    fn retry_reason(&self, attempt: &Attempt) -> Option<String> {
        match attempt {
            Attempt::Failed(error) => Some(error.to_string()),
            Attempt::Reply(reply) => {
                if !RETRYABLE.contains(&reply.status) {
                    return None;
                }
                if self.settings.drop_on_terminal_error && is_terminal(reply.status, &reply.text())
                {
                    return None;
                }
                Some(reply.status.to_string())
            }
        }
    }

    pub async fn upload(
        &self,
        object: &Object,
        body: &Bytes,
        limiter: &Limiter,
        in_flush: bool,
    ) -> Outcome {
        let url = object_url(&self.settings, &object.key);
        let headers = self.headers(object, body);
        let mut attempt_number = 0;
        let last = loop {
            let attempt = match self.attempt(&url, &headers, body).await {
                Ok(attempt) => attempt,
                Err(error) => {
                    tracing::error!("Error uploading to s3: {error}");
                    return Outcome::Retry;
                }
            };
            limiter.record(throttled(&attempt));
            if attempt_number + 1 >= ATTEMPTS {
                break attempt;
            }
            let Some(reason) = self.retry_reason(&attempt) else {
                break attempt;
            };
            if matches!(&attempt, Attempt::Reply(reply) if reply.status == 403) {
                self.forget_credentials().await;
            }
            let wait = (self.backoff)(attempt_number);
            if in_flush {
                tracing::debug!(
                    "S3 upload attempt {}/{ATTEMPTS} failed with {reason}, retrying in {:.2}s key={}",
                    attempt_number + 1,
                    wait.as_secs_f64(),
                    object.key
                );
            } else {
                tracing::warn!(
                    "S3 upload attempt {}/{ATTEMPTS} failed with {reason}, retrying in {:.2}s key={}",
                    attempt_number + 1,
                    wait.as_secs_f64(),
                    object.key
                );
            }
            self.retries.fetch_add(1, Ordering::Relaxed);
            tokio::time::sleep(wait).await;
            attempt_number += 1;
        };
        match last {
            Attempt::Reply(reply) if (200..300).contains(&reply.status) => Outcome::Delivered,
            Attempt::Reply(reply) => {
                let body = reply.text();
                tracing::error!(
                    "Error uploading to s3: status {} for {}: {}",
                    reply.status,
                    object.key,
                    body.chars().take(512).collect::<String>()
                );
                if self.settings.drop_on_terminal_error && is_terminal(reply.status, &body) {
                    tracing::warn!(
                        "s3 logging: dropping object {} after terminal status {}",
                        object.key,
                        reply.status
                    );
                    return Outcome::Dropped;
                }
                Outcome::Retry
            }
            Attempt::Failed(error) => {
                tracing::error!("Error uploading to s3: {error}");
                Outcome::Retry
            }
        }
    }

    pub async fn download(&self, key: &str) -> Option<Bytes> {
        let credentials = match self.credentials().await {
            Ok(credentials) => credentials,
            Err(error) => {
                tracing::error!("Error downloading from S3: {error}");
                return None;
            }
        };
        let Some(region) = self
            .settings
            .region_name
            .as_ref()
            .map(crate::pyjson::py_str)
        else {
            tracing::error!("Error downloading from S3: no s3_region_name to sign with");
            return None;
        };
        let headers = [("x-amz-content-sha256".to_owned(), EMPTY_SHA256.to_owned())];
        let reply = send(
            &self.download_client,
            SignedRequest {
                method: Method::GET,
                url: &object_url(&self.settings, key),
                headers: &headers,
                body: Bytes::new(),
                region: &region,
                credentials: &credentials,
                time: (self.clock)(),
                read_timeout: Some(self.settings.download_read_timeout),
            },
        )
        .await;
        match reply {
            Ok(reply) if reply.status == 200 => Some(reply.body),
            Ok(reply) => {
                tracing::error!("S3 object not found, saw response={}", reply.text());
                None
            }
            Err(error) => {
                tracing::error!("Error downloading from S3: {error}");
                None
            }
        }
    }
}

#[cfg(test)]
pub fn md5_base64(body: &[u8]) -> String {
    base64::engine::general_purpose::STANDARD.encode(<md5::Md5 as md5::Digest>::digest(body))
}

#[cfg(test)]
pub fn sha256_hex(body: &[u8]) -> String {
    crate::key::hex(&<sha2::Sha256 as sha2::Digest>::digest(body))
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    #[rstest]
    #[case(400, "<Error><Code>EntityTooLarge</Code></Error>", true)]
    #[case(403, "<Error><Code>BadDigest</Code></Error>", true)]
    #[case(403, "<Error><Code>AccessDenied</Code></Error>", false)]
    #[case(503, "<Error><Code>InvalidRequest</Code></Error>", false)]
    #[case(400, "bad request", false)]
    #[case(400, "<Code></Code><Code>InvalidDigest</Code>", true)]
    fn only_object_rejections_are_terminal(
        #[case] status: u16,
        #[case] body: &str,
        #[case] expected: bool,
    ) {
        assert_eq!(is_terminal(status, body), expected);
    }

    #[rstest]
    #[case(0, 0.5, 1.5)]
    #[case(1, 1.0, 3.0)]
    fn backoff_doubles_with_jitter(#[case] attempt: u32, #[case] low: f64, #[case] high: f64) {
        for _ in 0..50 {
            let wait = python_backoff(attempt).as_secs_f64();
            assert!((low..high).contains(&wait), "{wait}");
        }
    }

    #[rstest]
    fn digests_match_python() {
        assert_eq!(md5_base64(b"{}"), "mZFLkyvTelC5g8XnyQrpOw==");
        assert_eq!(
            sha256_hex(b""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
    }
}
