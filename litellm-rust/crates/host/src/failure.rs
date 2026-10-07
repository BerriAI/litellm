//! The failure contract every host reads: where a call failed and what went wrong.
//!
//! Routes report their own error type; the stage is attached by the shared call steps
//! ([`prepare`], [`receive`], [`post_call`], the provider send in `litellm_llms`) and by the
//! machine, never by route code. Hosts decide what to do from the stage and the [`Kind`] alone.

use std::{fmt, future::Future};

use serde::{Deserialize, Serialize};

use crate::machine::MachineFault;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Stage {
    /// Transforming the request, resolving credentials and secrets, running the pre-request
    /// hooks: nothing reached the provider.
    Prepare,
    /// Opening the connection failed: nothing reached the provider.
    Send,
    /// The provider answered with a non-success status.
    Upstream,
    /// The provider accepted the request; reading, streaming or decoding its answer failed.
    Receive,
    /// A hook that runs once the provider has answered failed.
    PostCall,
    /// The host driving the call failed: a hook raised, a reply was abandoned, or the
    /// protocol was broken.
    Host,
}

/// What went wrong, independent of the route that reports it.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Kind {
    /// The caller's request cannot be served as given.
    Request,
    /// The request asks for something this implementation does not provide.
    Unsupported,
    /// Credentials are missing or malformed.
    Auth,
    /// The provider did not answer in time.
    Timeout,
    /// The provider could not be reached, or the connection broke.
    Connection,
    /// The provider answered with a non-success status.
    Upstream(UpstreamResponse),
    /// The provider's answer could not be read or decoded.
    Response,
    /// A local file the request named could not be read.
    File { path: String, not_found: bool },
    /// Anything else.
    Internal,
}

/// A provider's non-success answer, as every host reports it.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct UpstreamResponse {
    pub status: u16,
    pub headers: Vec<(String, String)>,
    pub body: String,
    pub url: Option<String>,
}

impl fmt::Display for UpstreamResponse {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "upstream request failed with status {}: {}",
            self.status, self.body
        )
    }
}

impl std::error::Error for UpstreamResponse {}

/// A route error says what kind of failure it is; hosts never inspect its variants.
pub trait Classify {
    fn kind(&self) -> Kind;
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Failure<E> {
    pub stage: Stage,
    pub error: E,
}

/// The host-neutral description of a failure, as it crosses a language boundary.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Report {
    pub stage: Stage,
    pub kind: Kind,
    pub message: String,
}

impl<E> Failure<E> {
    /// A route error at a stage. Only the shared call steps and hosts call this; route code
    /// reaches a stage through [`prepare`], [`receive`], [`post_call`] and the provider send.
    pub fn at(stage: Stage, error: E) -> Self {
        Self { stage, error }
    }

    pub fn host(error: E) -> Self {
        Self::at(Stage::Host, error)
    }

    pub fn map<F>(self, map: impl FnOnce(E) -> F) -> Failure<F> {
        Failure {
            stage: self.stage,
            error: map(self.error),
        }
    }

    pub fn kind(&self) -> Kind
    where
        E: Classify,
    {
        self.error.kind()
    }

    pub fn report(&self) -> Report
    where
        E: Classify + fmt::Display,
    {
        Report {
            stage: self.stage,
            kind: self.kind(),
            message: self.to_string(),
        }
    }
}

impl<E: fmt::Display> fmt::Display for Failure<E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.error.fmt(f)
    }
}

impl<E: std::error::Error + 'static> std::error::Error for Failure<E> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.error)
    }
}

impl<E: From<MachineFault>> From<MachineFault> for Failure<E> {
    fn from(fault: MachineFault) -> Self {
        Self::host(E::from(fault))
    }
}

pub async fn prepare<T, E>(step: impl Future<Output = Result<T, E>>) -> Result<T, Failure<E>> {
    step.await
        .map_err(|error| Failure::at(Stage::Prepare, error))
}

pub async fn receive<T, E>(step: impl Future<Output = Result<T, E>>) -> Result<T, Failure<E>> {
    step.await
        .map_err(|error| Failure::at(Stage::Receive, error))
}

pub async fn post_call<T, E>(step: impl Future<Output = Result<T, E>>) -> Result<T, Failure<E>> {
    step.await
        .map_err(|error| Failure::at(Stage::PostCall, error))
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    #[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
    enum Plain {
        #[error("{0}")]
        Request(&'static str),
        #[error("{0}")]
        Upstream(UpstreamResponse),
    }

    impl Classify for Plain {
        fn kind(&self) -> Kind {
            match self {
                Self::Request(_) => Kind::Request,
                Self::Upstream(response) => Kind::Upstream(response.clone()),
            }
        }
    }

    fn upstream() -> UpstreamResponse {
        UpstreamResponse {
            status: 429,
            headers: vec![("retry-after".into(), "7".into())],
            body: "slow down".into(),
            url: Some("https://upstream.invalid/v1".into()),
        }
    }

    #[rstest]
    #[tokio::test]
    async fn each_step_stamps_its_own_stage() {
        let prepared: Result<(), Failure<Plain>> =
            prepare(async { Err(Plain::Request("p")) }).await;
        let received: Result<(), Failure<Plain>> =
            receive(async { Err(Plain::Request("r")) }).await;
        let hooked: Result<(), Failure<Plain>> =
            post_call(async { Err(Plain::Request("h")) }).await;
        assert_eq!(prepared.unwrap_err().stage, Stage::Prepare);
        assert_eq!(received.unwrap_err().stage, Stage::Receive);
        assert_eq!(hooked.unwrap_err().stage, Stage::PostCall);
        assert_eq!(Failure::host(Plain::Request("x")).stage, Stage::Host);
        assert_eq!(
            Failure::<Plain>::from(MachineFault::Abandoned).stage,
            Stage::Host
        );
    }

    impl From<MachineFault> for Plain {
        fn from(_: MachineFault) -> Self {
            Plain::Request("machine")
        }
    }

    #[rstest]
    #[tokio::test]
    async fn a_successful_step_passes_its_value_through() {
        let value: Result<u8, Failure<Plain>> = prepare(async { Ok(7) }).await;
        assert_eq!(value.unwrap(), 7);
    }

    #[rstest]
    fn the_report_carries_the_stage_the_kind_and_the_message() {
        let failure = Failure::at(Stage::Upstream, Plain::Upstream(upstream()));
        assert_eq!(
            failure.report(),
            Report {
                stage: Stage::Upstream,
                kind: Kind::Upstream(upstream()),
                message: "upstream request failed with status 429: slow down".into(),
            }
        );

        let failure = Failure::at(Stage::Prepare, Plain::Request("top_k"));
        assert_eq!(
            failure.report(),
            Report {
                stage: Stage::Prepare,
                kind: Kind::Request,
                message: "top_k".into(),
            }
        );
    }

    #[rstest]
    fn the_report_serializes_with_a_tagged_kind() {
        let report = Failure::at(Stage::Upstream, Plain::Upstream(upstream())).report();
        let json = serde_json::to_value(&report).unwrap();
        assert_eq!(json["stage"], "upstream");
        assert_eq!(json["kind"]["kind"], "upstream");
        assert_eq!(json["kind"]["status"], 429);
        assert_eq!(json["kind"]["headers"][0][0], "retry-after");
        assert_eq!(
            serde_json::from_value::<Report>(json).unwrap(),
            report,
            "a host decodes the same report it was sent"
        );
        let file = serde_json::to_value(Kind::File {
            path: "/scan.pdf".into(),
            not_found: true,
        })
        .unwrap();
        assert_eq!(
            file,
            serde_json::json!({"kind": "file", "path": "/scan.pdf", "not_found": true})
        );
        assert_eq!(
            serde_json::to_value(Kind::Request).unwrap(),
            serde_json::json!({"kind": "request"})
        );
    }

    #[rstest]
    fn map_keeps_the_stage() {
        let mapped =
            Failure::at(Stage::Receive, Plain::Request("x")).map(|error| error.to_string());
        assert_eq!(mapped, Failure::at(Stage::Receive, "x".to_string()));
    }
}
