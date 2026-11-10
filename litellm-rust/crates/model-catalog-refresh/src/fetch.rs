use std::borrow::Cow;
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::Duration;

use backon::Retryable;
use litellm_error::Report;
use litellm_http::Client;
use url::Url;

use crate::attempt::{self, Cause, Fetched, Rejected};
use crate::{RetryPolicy, Source};

/// How an `s3://` fetch reaches S3
#[derive(Clone, Debug)]
pub enum S3Access {
    /// What an AWS SDK finds in the environment and shared config files, loaded on each `s3://` attempt, with
    /// `region` winning over the region it finds
    Environment { region: Option<String> },
    /// A config the caller built with its region, endpoint and credentials
    Config(Box<litellm_s3::sdk::Config>),
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
    s3: &S3Access,
    url: &str,
    policy: RetryPolicy,
    start: Start,
) -> Result<Fetched, Failed> {
    let shown = redacted(url);
    let source =
        Source::parse(url).map_err(|error| Failed::of(&shown, &Rejected::fatal(error), 1))?;
    let mut backoff = policy.backoff();
    let made = AtomicU32::new(0);
    if let Start::After(failed) = start {
        let Some(wait) = backoff.next() else {
            return Err(failed);
        };
        tokio::time::sleep(policy.delay(wait, failed.retry_after)).await;
        made.store(1, Ordering::Relaxed);
    }
    let attempt = || {
        made.fetch_add(1, Ordering::Relaxed);
        async {
            match &source {
                Source::Http(http) => attempt::http(client, http.clone(), policy.timeout).await,
                Source::S3(object) => attempt::s3(client, s3, object, policy.timeout).await,
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
                made.load(Ordering::Relaxed),
                policy.max_attempts,
                reason(&shown, &rejected.cause),
                wait.as_secs_f64(),
            );
        })
        .await
        .map_err(|rejected| Failed::of(&shown, &rejected, made.load(Ordering::Relaxed)))
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
