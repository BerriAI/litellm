use std::sync::Arc;

use azure_core::error::ErrorKind;
use azure_core::http::headers::RETRY_AFTER;
use azure_core::http::policies::{Policy, PolicyResult};
use azure_core::http::{AsyncRawResponse, Context, Request, RetryOptions};
use azure_core::time::{Duration, OffsetDateTime};

use crate::retry_after::{RetryAfter, retry_after};
use crate::types::AzureTokenRetry;

const RETRIES_PER_KIND: u32 = 3;
const BACKOFF_FACTOR: Duration = Duration::milliseconds(800);
const BACKOFF_MAX: Duration = Duration::seconds(120);
const RETRY_STATUS_CODES: [u16; 6] = [408, 429, 500, 502, 503, 504];

pub(crate) enum TokenRetry {
    SdkDefault,
    Never,
    AzureIdentity,
}

impl TokenRetry {
    /// A client secret or workload identity exchange, which azure-identity retries in Python.
    pub(crate) fn exchange(policy: AzureTokenRetry) -> Self {
        match policy {
            AzureTokenRetry::SdkDefault => Self::SdkDefault,
            AzureTokenRetry::Python => Self::AzureIdentity,
        }
    }

    /// An OIDC assertion exchange, which Python posts once through LiteLLM's own HTTP handler.
    pub(crate) fn assertion(policy: AzureTokenRetry) -> Self {
        match policy {
            AzureTokenRetry::SdkDefault => Self::SdkDefault,
            AzureTokenRetry::Python => Self::Never,
        }
    }

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

/// The token endpoint's final answer, which a host replays through its own credential to raise
/// what Python raises for it.
#[derive(Debug)]
pub struct TokenResponse {
    pub status: u16,
    pub headers: Vec<(String, String)>,
    pub body: Vec<u8>,
}

impl TokenResponse {
    async fn read(response: AsyncRawResponse) -> azure_core::Result<Self> {
        let (status, headers, body) = response.deconstruct();
        Ok(Self {
            status: status.into(),
            headers: headers
                .iter()
                .map(|(name, value)| (name.as_str().to_string(), value.as_str().to_string()))
                .collect(),
            body: body.collect().await?.to_vec(),
        })
    }

    fn description(&self) -> String {
        serde_json::from_slice::<serde_json::Value>(&self.body)
            .ok()
            .and_then(|body| {
                body.get("error_description")?
                    .as_str()
                    .map(|text| format!(": {text}"))
            })
            .unwrap_or_default()
    }
}

/// How a token POST failed once azure-identity's retries are spent.
#[derive(Debug, thiserror::Error)]
pub enum EntraTokenFailure {
    #[error("Entra ID token endpoint answered {}{}", .0.status, .0.description())]
    Response(TokenResponse),
    #[error("Invalid HTTP date")]
    InvalidRetryAfter(TokenResponse),
    #[error("timestamp out of range for platform time_t")]
    RetryAfterTooLong(TokenResponse),
    #[error("Entra ID token request to {host}:{port} failed")]
    Transport {
        host: String,
        port: u16,
        #[source]
        source: azure_core::Error,
    },
}

impl EntraTokenFailure {
    pub fn response(&self) -> Option<&TokenResponse> {
        match self {
            Self::Response(response)
            | Self::InvalidRetryAfter(response)
            | Self::RetryAfterTooLong(response) => Some(response),
            Self::Transport { .. } => None,
        }
    }
}

enum Next {
    Wait(Duration),
    Rejected(fn(TokenResponse) -> EntraTokenFailure),
    Settle,
}

#[async_trait::async_trait]
impl Policy for AzureIdentityRetry {
    async fn send(
        &self,
        ctx: &Context,
        request: &mut Request,
        next: &[Arc<dyn Policy>],
    ) -> PolicyResult {
        let target = Target::of(request);
        let mut budget = RetryBudget::new();
        loop {
            let outcome = next[0].send(ctx, request, &next[1..]).await;
            let step = match &outcome {
                Ok(response) if is_retryable(response) && budget.spend(RetryKind::Status) => {
                    match retry_after(response.headers(), OffsetDateTime::now_utc()) {
                        RetryAfter::Backoff => Next::Wait(budget.backoff()),
                        RetryAfter::Wait(delay) => Next::Wait(delay),
                        RetryAfter::Unparseable => {
                            Next::Rejected(EntraTokenFailure::InvalidRetryAfter)
                        }
                        RetryAfter::TooLong => Next::Rejected(EntraTokenFailure::RetryAfterTooLong),
                    }
                }
                Err(error)
                    if transport_retry(error.kind()).is_some_and(|kind| budget.spend(kind)) =>
                {
                    Next::Wait(budget.backoff())
                }
                _ => Next::Settle,
            };
            match step {
                Next::Wait(delay) => azure_core::sleep::sleep(delay).await,
                Next::Rejected(failure) => return failed(&target, outcome?, failure).await,
                Next::Settle => return settle(&target, outcome).await,
            }
        }
    }
}

struct Target {
    host: String,
    port: u16,
}

impl Target {
    fn of(request: &Request) -> Self {
        Self {
            host: request.url().host_str().unwrap_or_default().to_string(),
            port: request.url().port_or_known_default().unwrap_or_default(),
        }
    }

    fn failed(&self, error: azure_core::Error) -> azure_core::Error {
        azure_core::Error::new(
            error.kind().clone(),
            EntraTokenFailure::Transport {
                host: self.host.clone(),
                port: self.port,
                source: error,
            },
        )
    }
}

async fn settle(target: &Target, outcome: PolicyResult) -> PolicyResult {
    match outcome {
        Ok(response) if u16::from(response.status()) >= 400 => {
            failed(target, response, EntraTokenFailure::Response).await
        }
        Err(error) if transport_retry(error.kind()).is_some() => Err(target.failed(error)),
        outcome => outcome,
    }
}

async fn failed(
    target: &Target,
    response: AsyncRawResponse,
    failure: fn(TokenResponse) -> EntraTokenFailure,
) -> PolicyResult {
    let status = response.status();
    let response = TokenResponse::read(response)
        .await
        .map_err(|error| target.failed(error))?;
    Err(azure_core::Error::new(
        ErrorKind::HttpResponse {
            status,
            error_code: None,
            raw_response: None,
        },
        failure(response),
    ))
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
