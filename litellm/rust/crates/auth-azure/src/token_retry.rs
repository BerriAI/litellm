use std::sync::Arc;

use azure_core::error::ErrorKind;
use azure_core::http::headers::{
    HeaderName, Headers, RETRY_AFTER, RETRY_AFTER_MS, X_MS_RETRY_AFTER_MS,
};
use azure_core::http::policies::{Policy, PolicyResult};
use azure_core::http::{AsyncRawResponse, Context, Request, RetryOptions};
use azure_core::time::{Duration, OffsetDateTime, parse_rfc7231};

const RETRIES_PER_KIND: u32 = 3;
const BACKOFF_FACTOR: Duration = Duration::milliseconds(800);
const BACKOFF_MAX: Duration = Duration::seconds(120);
const RETRY_STATUS_CODES: [u16; 6] = [408, 429, 500, 502, 503, 504];
const RETRY_AFTER_HEADERS: [(HeaderName, f64); 3] = [
    (RETRY_AFTER, 1.0),
    (RETRY_AFTER_MS, 0.001),
    (X_MS_RETRY_AFTER_MS, 0.001),
];

pub(crate) enum TokenRetry {
    SdkDefault,
    Never,
    AzureIdentity,
}

impl TokenRetry {
    pub(crate) fn into_options(self) -> (RetryOptions, Vec<Arc<dyn Policy>>) {
        match self {
            Self::SdkDefault => (RetryOptions::default(), Vec::new()),
            Self::Never => (RetryOptions::none(), Vec::new()),
            Self::AzureIdentity => (RetryOptions::none(), vec![Arc::new(AzureIdentityRetry)]),
        }
    }
}

/// azure-core's default `RetryPolicy` as azure-identity runs it for an Entra token POST
/// (`retry_on_methods=["POST"]`), so a token endpoint failure costs what it costs in Python.
#[derive(Debug)]
struct AzureIdentityRetry;

#[derive(Clone, Copy)]
enum RetryKind {
    Status,
    Connect,
    Read,
}

struct RetryBudget {
    status: u32,
    connect: u32,
    read: u32,
    retried: u32,
}

impl RetryBudget {
    fn new() -> Self {
        Self {
            status: RETRIES_PER_KIND,
            connect: RETRIES_PER_KIND,
            read: RETRIES_PER_KIND,
            retried: 0,
        }
    }

    fn spend(&mut self, kind: RetryKind) -> bool {
        let left = match kind {
            RetryKind::Status => &mut self.status,
            RetryKind::Connect => &mut self.connect,
            RetryKind::Read => &mut self.read,
        };
        if *left == 0 {
            return false;
        }
        *left -= 1;
        self.retried += 1;
        true
    }

    fn backoff(&self) -> Duration {
        if self.retried <= 1 {
            return Duration::ZERO;
        }
        let factor: i32 = 2_i32.saturating_pow(self.retried - 1);
        BACKOFF_FACTOR.saturating_mul(factor).min(BACKOFF_MAX)
    }
}

#[async_trait::async_trait]
impl Policy for AzureIdentityRetry {
    async fn send(
        &self,
        ctx: &Context,
        request: &mut Request,
        next: &[Arc<dyn Policy>],
    ) -> PolicyResult {
        let mut budget = RetryBudget::new();
        loop {
            let outcome = next[0].send(ctx, request, &next[1..]).await;
            let delay = match &outcome {
                Ok(response) if is_retryable(response) && budget.spend(RetryKind::Status) => {
                    retry_after(response.headers())?.unwrap_or_else(|| budget.backoff())
                }
                Err(error)
                    if transport_retry(error.kind()).is_some_and(|kind| budget.spend(kind)) =>
                {
                    budget.backoff()
                }
                _ => return outcome,
            };
            azure_core::sleep::sleep(delay).await;
        }
    }
}

fn is_retryable(response: &AsyncRawResponse) -> bool {
    let status = u16::from(response.status());
    let has_retry_after = response
        .headers()
        .get_optional_str(&RETRY_AFTER)
        .is_some_and(|value| !value.is_empty());
    status >= 400
        && (has_retry_after || RETRY_STATUS_CODES.contains(&status) || (506..999).contains(&status))
}

fn transport_retry(kind: &ErrorKind) -> Option<RetryKind> {
    match kind {
        ErrorKind::Connection => Some(RetryKind::Connect),
        ErrorKind::Io => Some(RetryKind::Read),
        _ => None,
    }
}

fn retry_after(headers: &Headers) -> azure_core::Result<Option<Duration>> {
    let Some((value, unit)) = RETRY_AFTER_HEADERS.iter().find_map(|(name, unit)| {
        headers
            .get_optional_str(name)
            .filter(|value| !value.is_empty())
            .map(|value| (value, *unit))
    }) else {
        return Ok(None);
    };
    parse_delay(value)
        .map(|delay| Some(delay * unit).filter(|delay| delay.is_positive()))
        .ok_or_else(|| {
            azure_core::Error::with_message(ErrorKind::DataConversion, "Invalid HTTP date")
        })
}

fn parse_delay(value: &str) -> Option<Duration> {
    value
        .trim()
        .parse::<f64>()
        .ok()
        .and_then(Duration::checked_seconds_f64)
        .or_else(|| {
            parse_rfc7231(value)
                .ok()
                .map(|date| date - OffsetDateTime::now_utc())
        })
        .map(|delay| delay.max(Duration::ZERO))
}
