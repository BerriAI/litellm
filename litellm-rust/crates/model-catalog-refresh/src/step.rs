use std::time::Duration;

use crate::Error;
use crate::policy::{RETRYABLE_STATUSES, RetryPolicy};

/// What one attempt got back
#[derive(Debug)]
pub enum Reply {
    /// An HTTP answer. The body is read for every status, since a non-error one is the document
    Response {
        status: u16,
        etag: Option<String>,
        /// `Retry-After` in seconds, when it parsed as a non-negative number
        retry_after: Option<f64>,
        body: Vec<u8>,
    },
    /// No usable answer. `retryable` is the driver's call, since only it knows the transport
    Transport { error: Error, retryable: bool },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Fetched {
    pub body: Vec<u8>,
    pub etag: Option<String>,
}

#[derive(Debug)]
pub enum Cause {
    Status(u16),
    Transport(Error),
}

#[derive(Debug)]
pub struct Failure {
    pub cause: Cause,
    pub attempts: u32,
    /// A retry could succeed, so the failure only ended the fetch because attempts ran out
    pub retryable: bool,
    pub retry_after: Option<f64>,
}

/// What the driver does next
#[derive(Debug)]
pub enum Step {
    /// Wait, then make the next attempt
    Sleep(Duration),
    Done(Result<Fetched, Failure>),
}

/// The fetch state machine: it counts attempts and decides between retrying and stopping, and does no I/O
#[derive(Clone, Copy, Debug)]
pub struct Attempts {
    policy: RetryPolicy,
    made: u32,
}

impl Attempts {
    pub fn new(policy: RetryPolicy) -> Self {
        Self { policy, made: 0 }
    }

    /// Continues a fetch whose first attempt failed retryably elsewhere, as main's background retry does after the
    /// import-time attempt. `None` when the policy allows no second attempt
    pub fn resume(
        policy: RetryPolicy,
        retry_after: Option<f64>,
        unit: f64,
    ) -> Option<(Self, Duration)> {
        (policy.max_attempts > 1)
            .then(|| (Self { policy, made: 1 }, policy.wait(1, retry_after, unit)))
    }

    pub fn timeout(&self) -> Duration {
        self.policy.timeout
    }

    /// Attempts made so far
    pub fn made(&self) -> u32 {
        self.made
    }

    pub fn max_attempts(&self) -> u32 {
        self.policy.max_attempts
    }

    /// The step after one attempt. `unit` is a uniform sample in `[0, 1)` for the backoff jitter
    pub fn next(&mut self, reply: Reply, unit: f64) -> Step {
        self.made += 1;
        let failure = match reply {
            Reply::Response {
                status, etag, body, ..
            } if status < 400 => return Step::Done(Ok(Fetched { body, etag })),
            Reply::Response {
                status,
                retry_after,
                ..
            } => Failure {
                cause: Cause::Status(status),
                attempts: self.made,
                retryable: RETRYABLE_STATUSES.contains(&status),
                retry_after,
            },
            Reply::Transport { error, retryable } => Failure {
                cause: Cause::Transport(error),
                attempts: self.made,
                retryable,
                retry_after: None,
            },
        };
        if failure.retryable && self.made < self.policy.max_attempts {
            return Step::Sleep(self.policy.wait(self.made, failure.retry_after, unit));
        }
        Step::Done(Err(failure))
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    const POLICY: RetryPolicy = RetryPolicy {
        max_attempts: 3,
        max_wait: Duration::from_secs(30),
        timeout: Duration::from_secs(5),
        backoff_base_seconds: 2.0,
        jitter_min_seconds: 0.0,
        jitter_max_seconds: 1.0,
    };

    fn status(status: u16, retry_after: Option<f64>) -> Reply {
        Reply::Response {
            status,
            etag: None,
            retry_after,
            body: Vec::new(),
        }
    }

    fn transport(retryable: bool) -> Reply {
        Reply::Transport {
            error: Error::unsupported_scheme("ftp".into()),
            retryable,
        }
    }

    #[rstest]
    fn a_success_returns_the_body_and_etag() {
        let mut attempts = Attempts::new(POLICY);
        let reply = Reply::Response {
            status: 200,
            etag: Some("\"v1\"".into()),
            retry_after: None,
            body: b"{}".to_vec(),
        };
        let Step::Done(Ok(fetched)) = attempts.next(reply, 0.0) else {
            panic!("a fetched document")
        };
        assert_eq!(
            fetched,
            Fetched {
                body: b"{}".to_vec(),
                etag: Some("\"v1\"".into())
            }
        );
    }

    #[rstest]
    fn a_429_waits_its_retry_after_then_succeeds() {
        let mut attempts = Attempts::new(POLICY);
        let Step::Sleep(wait) = attempts.next(status(429, Some(3.0)), 0.9) else {
            panic!("a retry")
        };
        assert_eq!(wait, Duration::from_secs(3));
        assert!(matches!(
            attempts.next(status(200, None), 0.0),
            Step::Done(Ok(_))
        ));
    }

    #[rstest]
    fn retryable_failures_stop_at_the_attempt_limit() {
        let mut attempts = Attempts::new(POLICY);
        assert_eq!(
            [503, 502].map(|code| match attempts.next(status(code, None), 0.5) {
                Step::Sleep(wait) => wait,
                step => panic!("a retry, got {step:?}"),
            }),
            [Duration::from_secs_f64(2.5), Duration::from_secs_f64(4.5)]
        );
        let Step::Done(Err(failure)) = attempts.next(status(500, None), 0.5) else {
            panic!("a failure")
        };
        assert!(matches!(failure.cause, Cause::Status(500)));
        assert_eq!((failure.attempts, failure.retryable), (3, true));
    }

    #[rstest]
    #[case::not_found(status(404, None))]
    #[case::forbidden(status(403, Some(1.0)))]
    #[case::final_transport(transport(false))]
    fn a_final_failure_never_retries(#[case] reply: Reply) {
        let mut attempts = Attempts::new(POLICY);
        let Step::Done(Err(failure)) = attempts.next(reply, 0.0) else {
            panic!("a failure")
        };
        assert_eq!((failure.attempts, failure.retryable), (1, false));
    }

    #[rstest]
    fn a_retryable_transport_error_retries() {
        let mut attempts = Attempts::new(POLICY);
        assert!(matches!(
            attempts.next(transport(true), 0.0),
            Step::Sleep(_)
        ));
    }

    #[rstest]
    fn a_redirect_is_returned_as_the_document() {
        let mut attempts = Attempts::new(POLICY);
        assert!(matches!(
            attempts.next(status(301, None), 0.0),
            Step::Done(Ok(_))
        ));
    }

    #[rstest]
    fn resume_waits_for_the_first_failure_then_makes_the_remaining_attempts() {
        let (mut attempts, wait) = Attempts::resume(POLICY, Some(4.0), 0.0).unwrap();
        assert_eq!(wait, Duration::from_secs(4));
        assert!(matches!(
            attempts.next(status(503, None), 0.0),
            Step::Sleep(_)
        ));
        let Step::Done(Err(failure)) = attempts.next(status(503, None), 0.0) else {
            panic!("a failure")
        };
        assert_eq!(failure.attempts, 3);
        assert!(
            Attempts::resume(
                RetryPolicy {
                    max_attempts: 1,
                    ..POLICY
                },
                None,
                0.0
            )
            .is_none()
        );
    }
}
