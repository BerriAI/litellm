use std::borrow::Cow;
use std::cell::Cell;
use std::fmt;
use std::time::Duration;

use backon::Retryable;
use litellm_error::Report;
use litellm_http::Client;
use litellm_s3::ProvideCredentials;
use url::Url;

use crate::attempt::{self, Cause, Fetched, Rejected};
use crate::{RetryPolicy, Source};

#[derive(Clone, PartialEq, Eq)]
pub struct Target {
    pub url: String,
    pub aws_region: String,
    pub aws_endpoint_url: Option<String>,
}

impl fmt::Debug for Target {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Target")
            .field("url", &redacted(&self.url))
            .field("aws_region", &self.aws_region)
            .field("aws_endpoint_url", &self.aws_endpoint_url)
            .finish()
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Failed {
    pub reason: String,
    /// Another attempt could succeed. Only the attempt limit ended the fetch
    pub retryable: bool,
    pub retry_after: Option<Duration>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Start {
    Fresh,
    /// Continues a fetch whose first attempt failed retryably elsewhere, waiting before its next attempt
    After(Failed),
}

pub async fn fetch(
    client: &Client,
    credentials: &(impl ProvideCredentials + Clone + 'static),
    target: &Target,
    policy: RetryPolicy,
    start: Start,
) -> Result<Fetched, Failed> {
    let url = redacted(&target.url);
    let source =
        Source::parse(&target.url).map_err(|error| Failed::of(&url, &Rejected::fatal(error), 1))?;
    let mut backoff = policy.backoff();
    let made = Cell::new(0);
    if let Start::After(failed) = start {
        let Some(wait) = backoff.next() else {
            return Err(failed);
        };
        tokio::time::sleep(policy.delay(wait, failed.retry_after)).await;
        made.set(1);
    }
    let attempt = || {
        made.set(made.get() + 1);
        async {
            match &source {
                Source::Http(http) => attempt::http(client, http.clone(), policy.timeout).await,
                Source::S3(object) => {
                    attempt::s3(client, credentials, target, object, policy.timeout).await
                }
            }
        }
    };
    attempt
        .retry(backoff)
        .sleep(tokio::time::sleep)
        .when(|rejected| rejected.retryable)
        .adjust(|rejected, wait| wait.map(|wait| policy.delay(wait, rejected.retry_after)))
        .notify(|rejected, wait| {
            litellm_tracing::warn!(
                "LiteLLM: model cost map fetch attempt {}/{} failed ({}); retrying in {:.1}s",
                made.get(),
                policy.max_attempts,
                reason(&url, &rejected.cause),
                wait.as_secs_f64(),
            );
        })
        .await
        .map_err(|rejected| Failed::of(&url, &rejected, made.get()))
}

impl Failed {
    fn of(url: &str, rejected: &Rejected, attempts: u32) -> Self {
        let reason = reason(url, &rejected.cause);
        Self {
            reason: match rejected.retryable && attempts > 1 {
                true => format!("{reason} (after {attempts} attempts)"),
                false => reason,
            },
            retryable: rejected.retryable,
            retry_after: rejected.retry_after,
        }
    }
}

fn reason(url: &str, cause: &Cause) -> String {
    match cause {
        Cause::Status(status) => format!("HTTP {status} from {url}"),
        Cause::Transport(error) => format!("error fetching {url}: {}", Report::new(error)),
    }
}

/// Drops userinfo, query and fragment, where presigned URLs and tokens carry credentials
fn redacted(url: &str) -> Cow<'_, str> {
    let Ok(mut parsed) = Url::parse(url) else {
        return Cow::Borrowed(url);
    };
    if parsed.username().is_empty()
        && parsed.password().is_none()
        && parsed.query().is_none()
        && parsed.fragment().is_none()
    {
        return Cow::Borrowed(url);
    }
    let _ = parsed.set_username("");
    let _ = parsed.set_password(None);
    parsed.set_query(None);
    parsed.set_fragment(None);
    Cow::Owned(parsed.into())
}
