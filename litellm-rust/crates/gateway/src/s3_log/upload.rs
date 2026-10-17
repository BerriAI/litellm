//! One signed PUT per object, retried as the proxy's `_put_with_retries` retries it: three
//! attempts, a transport error or a 403, 429, 500 or 503 retried after a jittered backoff, and
//! a 400 or 403 that names an object-specific error code dropped instead of kept for later.

use std::{
    collections::BTreeMap,
    sync::Arc,
    time::{Duration, Instant, SystemTime},
};

use base64::Engine;
use litellm_auth_aws::{AwsAuthService, Credentials, sign_s3};
use litellm_http::{ClientVariant, HttpClientConfig, HttpClientPool, Verify};
use md5::Md5;
use sha2::{Digest, Sha256};

use super::{Element, config::Settings, key::object_url};

const ATTEMPTS: u32 = 3;
const RETRYABLE: [u16; 4] = [403, 429, 500, 503];
/// Codes S3 answers that mean this object can never be stored, so retrying it is pointless.
const TERMINAL_CODES: [&str; 7] = [
    "EntityTooLarge",
    "InvalidArgument",
    "MalformedXML",
    "InvalidDigest",
    "KeyTooLongError",
    "BadDigest",
    "InvalidRequest",
];
/// Credentials without an expiry are resolved again after this long.
const CREDENTIALS_TTL: Duration = Duration::from_secs(600);
/// Credentials are refreshed this long before they expire.
const CREDENTIALS_MARGIN: Duration = Duration::from_secs(300);

/// How one object's upload ended.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum Upload {
    Delivered,
    /// Failed for now; it goes back on the queue.
    Retry,
    /// S3 rejected the object itself.
    Dropped,
}

/// What one attempt got back.
enum Attempt {
    Response { status: u16, body: String },
    Transport(String),
}

impl Attempt {
    fn is_terminal(&self) -> bool {
        matches!(self, Self::Response { status, body } if is_terminal(*status, body))
    }
}

/// A 400 or 403 whose `<Code>` S3 reserves for a rejection of this object.
fn is_terminal(status: u16, body: &str) -> bool {
    matches!(status, 400 | 403)
        && error_code(body).is_some_and(|code| TERMINAL_CODES.contains(&code))
}

fn error_code(body: &str) -> Option<&str> {
    let start = body.find("<Code>")? + "<Code>".len();
    let length = body[start..].find("</Code>")?;
    Some(&body[start..start + length]).filter(|code| !code.is_empty() && !code.contains('<'))
}

/// `(1 << attempt) * (0.5 + random())` seconds.
fn backoff(attempt: u32) -> Duration {
    Duration::from_secs_f64(f64::from(1u32 << attempt) * (0.5 + rand::random::<f64>()))
}

pub(super) struct Uploader {
    settings: Arc<Settings>,
    pool: Arc<HttpClientPool>,
    http: HttpClientConfig,
    auth: AwsAuthService,
    credentials: tokio::sync::Mutex<Option<(Credentials, Instant)>>,
}

impl Uploader {
    pub fn new(
        settings: Arc<Settings>,
        pool: Arc<HttpClientPool>,
        http: &HttpClientConfig,
    ) -> Self {
        let verify = match &settings.verify {
            None => http.verify.clone(),
            Some(litellm_http::SslVerify::Enabled) => Verify::BuiltInRoots,
            Some(litellm_http::SslVerify::Disabled) => Verify::Disabled,
            Some(litellm_http::SslVerify::CaBundle(path)) => Verify::CaBundle(path.clone()),
        };
        Self {
            settings,
            pool,
            http: HttpClientConfig {
                verify,
                ..http.clone()
            },
            auth: AwsAuthService::default(),
            credentials: tokio::sync::Mutex::new(None),
        }
    }

    pub async fn upload(&self, element: &Element) -> Upload {
        let url = object_url(&self.settings, &element.key);
        let headers = self.headers(element);
        let mut attempt = 0;
        loop {
            let result = match self.put(&url, element, &headers).await {
                Ok(result) => result,
                Err(error) => {
                    tracing::error!(%error, key = %element.key, "s3 logging: cannot sign the upload");
                    return Upload::Retry;
                }
            };
            let retry = match &result {
                Attempt::Transport(_) => true,
                Attempt::Response { status, .. } => {
                    RETRYABLE.contains(status)
                        && !(self.settings.drop_on_terminal_error && result.is_terminal())
                }
            };
            if retry && attempt + 1 < ATTEMPTS {
                let wait = backoff(attempt);
                let reason = match &result {
                    Attempt::Transport(error) => error.clone(),
                    Attempt::Response { status, .. } => status.to_string(),
                };
                tracing::debug!(attempt = attempt + 1, %reason, ?wait, key = %element.key, "s3 logging: upload failed, retrying");
                tokio::time::sleep(wait).await;
                attempt += 1;
                continue;
            }
            return match result {
                Attempt::Response { status, .. } if (200..300).contains(&status) => {
                    Upload::Delivered
                }
                Attempt::Response { status, body } => {
                    tracing::error!(status, body = %body.chars().take(512).collect::<String>(), key = %element.key, "s3 logging: upload failed");
                    if self.settings.drop_on_terminal_error && is_terminal(status, &body) {
                        tracing::warn!(status, key = %element.key, "s3 logging: dropping the object, S3 rejected it");
                        return Upload::Dropped;
                    }
                    Upload::Retry
                }
                Attempt::Transport(error) => {
                    tracing::error!(%error, key = %element.key, "s3 logging: upload failed");
                    Upload::Retry
                }
            };
        }
    }

    /// The proxy's PUT headers, before signing.
    fn headers(&self, element: &Element) -> BTreeMap<String, String> {
        let body = &element.body;
        let sha256: String = Sha256::digest(body)
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect();
        let md5 =
            base64::engine::general_purpose::STANDARD.encode(<Md5 as md5::Digest>::digest(body));
        let mut headers = BTreeMap::from([
            ("Content-Type".to_owned(), element.content_type.to_owned()),
            ("Content-MD5".to_owned(), md5),
            ("x-amz-content-sha256".to_owned(), sha256),
            ("Content-Language".to_owned(), "en".to_owned()),
            (
                "Content-Disposition".to_owned(),
                format!("inline; filename=\"{}\"", element.download_name),
            ),
            (
                "Cache-Control".to_owned(),
                "private, immutable, max-age=31536000, s-maxage=0".to_owned(),
            ),
        ]);
        if let Some(algorithm) = &self.settings.server_side_encryption {
            headers.insert("x-amz-server-side-encryption".into(), algorithm.clone());
        }
        if let Some(key_id) = &self.settings.sse_kms_key_id {
            headers.insert(
                "x-amz-server-side-encryption-aws-kms-key-id".into(),
                key_id.clone(),
            );
        }
        headers
    }

    async fn put(
        &self,
        url: &str,
        element: &Element,
        headers: &BTreeMap<String, String>,
    ) -> Result<Attempt, Box<dyn std::error::Error + Send + Sync>> {
        let credentials = self.credentials().await?;
        let signature = sign_s3(
            "PUT",
            url,
            &element.body,
            headers,
            &self.settings.region,
            &credentials,
            SystemTime::now(),
        )?;
        let client = self.pool.client(&self.http, ClientVariant::Provider)?;
        let mut request = client
            .put(url)
            .timeout(self.settings.attempt_timeout)
            .body(element.body.clone());
        for (name, value) in headers.iter().chain(&signature) {
            request = request.header(name, value);
        }
        Ok(match request.send().await {
            Ok(response) => Attempt::Response {
                status: response.status().as_u16(),
                body: response.text().await.unwrap_or_default(),
            },
            Err(error) => Attempt::Transport(error.to_string()),
        })
    }

    async fn credentials(&self) -> Result<Credentials, litellm_auth_aws::Error> {
        let mut cached = self.credentials.lock().await;
        if let Some((credentials, until)) = cached.as_ref()
            && Instant::now() < *until
        {
            return Ok(credentials.clone());
        }
        let env = |name: &str| std::env::var(name).ok();
        let credentials = self
            .auth
            .resolve_credentials(self.settings.auth.clone(), &env)
            .await?;
        let until = credentials
            .expiry()
            .and_then(|expiry| expiry.duration_since(SystemTime::now()).ok())
            .map_or(CREDENTIALS_TTL, |left| {
                left.saturating_sub(CREDENTIALS_MARGIN)
            });
        *cached = Some((credentials.clone(), Instant::now() + until));
        Ok(credentials)
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    #[rstest]
    #[case::too_large(400, "<Error><Code>EntityTooLarge</Code></Error>", true)]
    #[case::bad_digest(403, "<Error><Code>BadDigest</Code></Error>", true)]
    #[case::access_denied(403, "<Error><Code>AccessDenied</Code></Error>", false)]
    #[case::slow_down(503, "<Error><Code>InvalidRequest</Code></Error>", false)]
    #[case::not_xml(400, "bad request", false)]
    #[case::empty_code(400, "<Code></Code>", false)]
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
            let wait = backoff(attempt).as_secs_f64();
            assert!((low..high).contains(&wait), "{wait}");
        }
    }
}
