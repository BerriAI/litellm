use crate::retry::{AfterRetryable, after_retryable};
use crate::status::{Attempt, classify};
use crate::{Fetch, FetchRequest};

/// How a refresh ended
#[derive(Clone, Debug, PartialEq)]
pub enum Refreshed {
    Adopted {
        body: Vec<u8>,
        /// `git_blob_id` of `body`
        revision: String,
        etag: Option<String>,
    },
    NotModified,
    Failed {
        reason: String,
    },
}

/// A retryable failure about to be waited out
#[derive(Clone, Debug, PartialEq)]
pub struct Retrying<'a> {
    pub attempt: i64,
    pub max_attempts: i64,
    pub reason: &'a str,
    pub wait_seconds: f64,
}

/// What a refresh does between attempts, injected so tests and hosts choose their own clock and randomness. A host
/// whose jitter or report can fail returns that failure from the `sleep` that follows it, which ends the refresh
pub trait Pause: Send {
    type Stop;

    /// Seconds added to a backoff, drawn only when there is no usable `Retry-After`
    fn jitter(&mut self) -> f64;
    fn retrying(&mut self, retrying: &Retrying<'_>);
    fn sleep(&mut self, seconds: f64) -> impl Future<Output = Result<(), Self::Stop>> + Send;
}

/// Up to `max_attempts` attempts, as main's `_fetch_remote_model_cost_map_with_retry`
pub async fn refresh<P: Pause>(
    fetch: &impl Fetch,
    request: &FetchRequest,
    max_attempts: i64,
    pause: &mut P,
) -> Result<Refreshed, P::Stop> {
    if max_attempts < 1 {
        return Ok(Refreshed::Failed {
            reason: "model cost map fetch failed".to_owned(),
        });
    }
    let first = attempt(fetch, request).await;
    resume(fetch, request, 1, max_attempts, first, pause).await
}

/// One attempt, for a host that makes the first attempt itself and resumes only after a retryable one
pub async fn attempt(fetch: &impl Fetch, request: &FetchRequest) -> Attempt {
    classify(fetch.fetch(request).await, fetch.location(), request)
}

/// Carries on after `outcome`, the result of attempt number `attempt`
pub async fn resume<P: Pause>(
    fetch: &impl Fetch,
    request: &FetchRequest,
    attempt: i64,
    max_attempts: i64,
    outcome: Attempt,
    pause: &mut P,
) -> Result<Refreshed, P::Stop> {
    let (mut attempt_number, mut outcome) = (attempt, outcome);
    loop {
        let (reason, retry_after) = match outcome {
            Attempt::Fetched { body, etag } => {
                return Ok(Refreshed::Adopted {
                    revision: litellm_model_catalog::revision(&body),
                    body,
                    etag,
                });
            }
            Attempt::NotModified => return Ok(Refreshed::NotModified),
            Attempt::Failed { reason } => return Ok(Refreshed::Failed { reason }),
            Attempt::Retryable {
                reason,
                retry_after,
            } => (reason, retry_after),
        };
        let seconds =
            match after_retryable(attempt_number, max_attempts, retry_after, || pause.jitter()) {
                AfterRetryable::GiveUp => {
                    return Ok(Refreshed::Failed {
                        reason: format!("{reason} (after {max_attempts} attempts)"),
                    });
                }
                AfterRetryable::Wait { seconds } => seconds,
            };
        pause.retrying(&Retrying {
            attempt: attempt_number,
            max_attempts,
            reason: &reason,
            wait_seconds: seconds,
        });
        pause.sleep(seconds).await?;
        attempt_number += 1;
        outcome = self::attempt(fetch, request).await;
    }
}

#[cfg(test)]
mod tests {
    use std::collections::VecDeque;
    use std::io;
    use std::sync::Mutex;
    use std::time::Duration;

    use rstest::rstest;

    use super::*;
    use crate::Reply;

    const URL: &str = "https://example.com/map.json";

    /// Answers each attempt from a script and records the requests it saw
    struct Scripted {
        replies: Mutex<VecDeque<Reply>>,
        requests: Mutex<Vec<FetchRequest>>,
    }

    impl Scripted {
        fn new(replies: impl IntoIterator<Item = Reply>) -> Self {
            Self {
                replies: Mutex::new(replies.into_iter().collect()),
                requests: Mutex::new(Vec::new()),
            }
        }

        fn attempts(&self) -> usize {
            self.requests.lock().unwrap().len()
        }
    }

    impl Fetch for Scripted {
        fn location(&self) -> &str {
            URL
        }

        async fn fetch(&self, request: &FetchRequest) -> Reply {
            self.requests.lock().unwrap().push(request.clone());
            self.replies
                .lock()
                .unwrap()
                .pop_front()
                .expect("a scripted reply")
        }
    }

    #[derive(Default)]
    struct Recording {
        jitters: usize,
        slept: Vec<f64>,
        retried: Vec<(i64, i64, String, f64)>,
    }

    impl Pause for Recording {
        type Stop = std::convert::Infallible;

        fn jitter(&mut self) -> f64 {
            self.jitters += 1;
            0.5
        }

        async fn sleep(&mut self, seconds: f64) -> Result<(), Self::Stop> {
            self.slept.push(seconds);
            Ok(())
        }

        fn retrying(&mut self, retrying: &Retrying<'_>) {
            self.retried.push((
                retrying.attempt,
                retrying.max_attempts,
                retrying.reason.to_owned(),
                retrying.wait_seconds,
            ));
        }
    }

    fn request() -> FetchRequest {
        FetchRequest {
            if_none_match: None,
            timeout: Duration::from_secs(5),
        }
    }

    fn status(status: u16, retry_after: Option<&str>) -> Reply {
        Reply::Response {
            status,
            etag: None,
            retry_after: retry_after.map(str::to_owned),
            body: Vec::new(),
        }
    }

    fn ok(body: &[u8]) -> Reply {
        Reply::Response {
            status: 200,
            etag: Some("\"v2\"".to_owned()),
            retry_after: None,
            body: body.to_vec(),
        }
    }

    fn reset() -> Reply {
        Reply::Transport {
            error: io::Error::from(io::ErrorKind::ConnectionReset).into(),
        }
    }

    #[rstest]
    #[tokio::test]
    async fn retries_then_adopts_with_the_revision_of_the_body() {
        let fetch = Scripted::new([status(503, Some("1")), status(503, None), ok(b"hello\n")]);
        let mut pause = Recording::default();
        let refreshed = refresh(&fetch, &request(), 3, &mut pause).await.unwrap();
        assert_eq!(
            refreshed,
            Refreshed::Adopted {
                body: b"hello\n".to_vec(),
                revision: "ce013625030ba8dba906f756967f9e9ca394464a".to_owned(),
                etag: Some("\"v2\"".to_owned()),
            }
        );
        assert_eq!(pause.slept, vec![1.0, 4.5]);
        assert_eq!(pause.jitters, 1);
        assert_eq!(
            pause.retried,
            vec![
                (1, 3, format!("HTTP 503 from {URL}"), 1.0),
                (2, 3, format!("HTTP 503 from {URL}"), 4.5),
            ]
        );
    }

    #[rstest]
    #[tokio::test]
    async fn gives_up_after_the_last_attempt_without_sleeping() {
        let fetch = Scripted::new([status(503, Some("0")), reset(), status(429, Some("0"))]);
        let mut pause = Recording::default();
        let refreshed = refresh(&fetch, &request(), 3, &mut pause).await.unwrap();
        assert_eq!(
            refreshed,
            Refreshed::Failed {
                reason: format!("HTTP 429 from {URL} (after 3 attempts)"),
            }
        );
        assert_eq!(fetch.attempts(), 3);
        assert_eq!(pause.slept.len(), 2);
    }

    #[rstest]
    #[tokio::test]
    async fn a_final_status_stops_at_once() {
        let fetch = Scripted::new([status(404, None)]);
        let mut pause = Recording::default();
        let refreshed = refresh(&fetch, &request(), 3, &mut pause).await.unwrap();
        assert_eq!(
            refreshed,
            Refreshed::Failed {
                reason: format!("HTTP 404 from {URL}"),
            }
        );
        assert!(pause.slept.is_empty());
    }

    #[rstest]
    #[case::zero(0)]
    #[case::negative(-1)]
    #[tokio::test]
    async fn no_attempts_fetch_nothing(#[case] max_attempts: i64) {
        let fetch = Scripted::new([]);
        let refreshed = refresh(&fetch, &request(), max_attempts, &mut Recording::default())
            .await
            .unwrap();
        assert_eq!(
            refreshed,
            Refreshed::Failed {
                reason: "model cost map fetch failed".to_owned(),
            }
        );
        assert_eq!(fetch.attempts(), 0);
    }

    #[rstest]
    #[tokio::test]
    async fn resume_waits_out_the_first_attempt_then_counts_from_two() {
        let fetch = Scripted::new([ok(b"{}")]);
        let mut pause = Recording::default();
        let first = Attempt::Retryable {
            reason: "HTTP 503 from x".to_owned(),
            retry_after: Some(2.0),
        };
        let refreshed = resume(&fetch, &request(), 1, 3, first, &mut pause)
            .await
            .unwrap();
        assert!(matches!(refreshed, Refreshed::Adopted { .. }));
        assert_eq!(
            pause.retried,
            vec![(1, 3, "HTTP 503 from x".to_owned(), 2.0)]
        );
        assert_eq!(fetch.attempts(), 1);
    }

    /// Stops at the first sleep, as a host whose jitter or report failed does
    struct Stopping;

    impl Pause for Stopping {
        type Stop = &'static str;

        fn jitter(&mut self) -> f64 {
            0.0
        }

        fn retrying(&mut self, _: &Retrying<'_>) {}

        async fn sleep(&mut self, _: f64) -> Result<(), Self::Stop> {
            Err("stopped")
        }
    }

    #[rstest]
    #[tokio::test]
    async fn a_stopped_pause_ends_the_refresh_before_the_next_attempt() {
        let fetch = Scripted::new([status(503, None), ok(b"{}")]);
        let stopped = refresh(&fetch, &request(), 3, &mut Stopping).await;
        assert_eq!(stopped, Err("stopped"));
        assert_eq!(fetch.attempts(), 1);
    }

    #[rstest]
    #[tokio::test]
    async fn a_conditional_not_modified_ends_the_refresh() {
        let fetch = Scripted::new([status(304, None)]);
        let request = FetchRequest {
            if_none_match: Some("\"v1\"".to_owned()),
            ..request()
        };
        let refreshed = refresh(&fetch, &request, 3, &mut Recording::default())
            .await
            .unwrap();
        assert_eq!(refreshed, Refreshed::NotModified);
        assert_eq!(
            fetch.requests.lock().unwrap()[0].if_none_match.as_deref(),
            Some("\"v1\"")
        );
    }
}
