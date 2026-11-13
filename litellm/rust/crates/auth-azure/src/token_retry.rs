use std::sync::Arc;

use azure_core::http::headers::{RETRY_AFTER, RETRY_AFTER_MS, X_MS_RETRY_AFTER_MS};
use azure_core::http::policies::{RetryHeaders, RetryPolicy};
use azure_core::http::{RetryOptions, StatusCode};
use azure_core::time::Duration;

const MAX_RETRIES: u32 = 3;
const BACKOFF_FACTOR: Duration = Duration::milliseconds(800);
const BACKOFF_MAX: Duration = Duration::seconds(120);

/// azure-core's default `RetryPolicy` as azure-identity runs it for an Entra token POST
/// (`retry_on_methods=["POST"]`), so a token endpoint failure costs what it costs in Python.
#[derive(Debug)]
struct EntraTokenRetryPolicy {
    retry_headers: RetryHeaders,
    retry_status_codes: Vec<StatusCode>,
}

impl Default for EntraTokenRetryPolicy {
    fn default() -> Self {
        Self {
            retry_headers: RetryHeaders {
                retry_headers: vec![RETRY_AFTER, RETRY_AFTER_MS, X_MS_RETRY_AFTER_MS],
            },
            retry_status_codes: [408, 429, 500, 502, 503, 504]
                .into_iter()
                .chain(506..999)
                .map(StatusCode::from)
                .collect(),
        }
    }
}

#[async_trait::async_trait]
impl RetryPolicy for EntraTokenRetryPolicy {
    fn is_expired(&self, _duration_since_start: Duration, retry_count: u32) -> bool {
        retry_count >= MAX_RETRIES
    }

    fn retry_headers(&self) -> Option<&RetryHeaders> {
        Some(&self.retry_headers)
    }

    fn retry_status_codes(&self) -> &[StatusCode] {
        &self.retry_status_codes
    }

    fn sleep_duration(&self, retry_count: u32) -> Duration {
        if retry_count <= 1 {
            return Duration::ZERO;
        }
        let factor: i32 = 2_i32.saturating_pow(retry_count - 1);
        BACKOFF_FACTOR.saturating_mul(factor).min(BACKOFF_MAX)
    }

    async fn wait(&self, retry_count: u32, retry_after: Option<Duration>) {
        let delay = retry_after
            .filter(|delay| delay.is_positive())
            .unwrap_or_else(|| self.sleep_duration(retry_count));
        azure_core::sleep::sleep(delay).await;
    }
}

pub(crate) fn entra_token_retry() -> RetryOptions {
    RetryOptions::custom(Arc::new(EntraTokenRetryPolicy::default()))
}
