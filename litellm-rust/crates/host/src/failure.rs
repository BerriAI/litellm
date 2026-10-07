//! The failure contract every host reads: where a call failed and what went wrong.
//!
//! Routes report their own error type and never pick a stage. A stage is attached by the
//! shared steps ([`prepare`], [`receive`], [`post_call`]), by an [`Exchange`] with the
//! provider, or by the machine for the op a host was answering. Hosts decide what to do
//! from the facts on [`Failure`] and [`Report`], never from stage or kind matches.

use std::{fmt, future::Future};

use serde::{Deserialize, Serialize};

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
    /// The provider accepted the request; reading, streaming, decoding or delivering its
    /// answer failed.
    Receive,
    /// A hook that runs once the provider has answered failed.
    PostCall,
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
    /// The implementation itself failed.
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

/// How an exchange with the provider ended without a success response.
pub enum Exchange<E> {
    /// The connection never opened: nothing reached the provider.
    Unreached(E),
    /// The provider answered with a non-success status.
    Rejected(UpstreamResponse),
    /// The request went out and the answer never fully arrived.
    Broken(E),
}

impl<E> Failure<E> {
    pub(crate) fn at(stage: Stage, error: E) -> Self {
        Self { stage, error }
    }

    pub fn prepare(error: E) -> Self {
        Self::at(Stage::Prepare, error)
    }

    pub fn receive(error: E) -> Self {
        Self::at(Stage::Receive, error)
    }

    pub fn post_call(error: E) -> Self {
        Self::at(Stage::PostCall, error)
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

    /// Nothing reached the provider.
    pub fn provider_untouched(&self) -> bool {
        provider_untouched(self.stage)
    }

    /// The provider saw the request and turned it down as given.
    pub fn rejected_by_provider(&self) -> bool
    where
        E: Classify,
    {
        rejected_by_provider(self.stage, &self.kind())
    }

    /// The same request may succeed later on the same path.
    pub fn is_retryable(&self) -> bool
    where
        E: Classify,
    {
        is_retryable(&self.kind())
    }

    /// Another implementation of the same request may serve it without the provider
    /// seeing a duplicate side effect.
    pub fn is_reroutable(&self) -> bool
    where
        E: Classify,
    {
        self.provider_untouched() || self.rejected_by_provider()
    }
}

impl<E> From<Exchange<E>> for Failure<E>
where
    E: From<UpstreamResponse>,
{
    fn from(exchange: Exchange<E>) -> Self {
        match exchange {
            Exchange::Unreached(error) => Self::at(Stage::Send, error),
            Exchange::Rejected(response) => Self::at(Stage::Upstream, E::from(response)),
            Exchange::Broken(error) => Self::at(Stage::Receive, error),
        }
    }
}

impl Report {
    pub fn provider_untouched(&self) -> bool {
        provider_untouched(self.stage)
    }

    pub fn rejected_by_provider(&self) -> bool {
        rejected_by_provider(self.stage, &self.kind)
    }

    pub fn is_retryable(&self) -> bool {
        is_retryable(&self.kind)
    }

    pub fn is_reroutable(&self) -> bool {
        self.provider_untouched() || self.rejected_by_provider()
    }
}

fn provider_untouched(stage: Stage) -> bool {
    matches!(stage, Stage::Prepare | Stage::Send)
}

fn rejected_by_provider(stage: Stage, kind: &Kind) -> bool {
    match kind {
        Kind::Upstream(response) if stage == Stage::Upstream => {
            (400..500).contains(&response.status) && !is_retryable(kind)
        }
        _ => false,
    }
}

fn is_retryable(kind: &Kind) -> bool {
    match kind {
        Kind::Timeout | Kind::Connection => true,
        Kind::Upstream(response) => matches!(response.status, 408 | 429 | 500..=599),
        Kind::Request
        | Kind::Unsupported
        | Kind::Auth
        | Kind::Response
        | Kind::Internal => false,
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

pub async fn prepare<T, E>(step: impl Future<Output = Result<T, E>>) -> Result<T, Failure<E>> {
    step.await.map_err(Failure::prepare)
}

pub async fn receive<T, E>(step: impl Future<Output = Result<T, E>>) -> Result<T, Failure<E>> {
    step.await.map_err(Failure::receive)
}

pub async fn post_call<T, E>(step: impl Future<Output = Result<T, E>>) -> Result<T, Failure<E>> {
    step.await.map_err(Failure::post_call)
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
        Timeout(&'static str),
        #[error("{0}")]
        Upstream(UpstreamResponse),
    }

    impl Classify for Plain {
        fn kind(&self) -> Kind {
            match self {
                Self::Request(_) => Kind::Request,
                Self::Timeout(_) => Kind::Timeout,
                Self::Upstream(response) => Kind::Upstream(response.clone()),
            }
        }
    }

    impl From<UpstreamResponse> for Plain {
        fn from(response: UpstreamResponse) -> Self {
            Self::Upstream(response)
        }
    }

    fn upstream(status: u16) -> UpstreamResponse {
        UpstreamResponse {
            status,
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
    }

    #[rstest]
    #[case::unreached(Exchange::Unreached(Plain::Request("refused")), Stage::Send)]
    #[case::rejected(Exchange::Rejected(upstream(400)), Stage::Upstream)]
    #[case::broken(Exchange::Broken(Plain::Timeout("slow")), Stage::Receive)]
    fn an_exchange_ends_at_the_stage_it_reached(
        #[case] exchange: Exchange<Plain>,
        #[case] stage: Stage,
    ) {
        assert_eq!(Failure::from(exchange).stage, stage);
    }

    #[rstest]
    #[tokio::test]
    async fn a_successful_step_passes_its_value_through() {
        let value: Result<u8, Failure<Plain>> = prepare(async { Ok(7) }).await;
        assert_eq!(value.unwrap(), 7);
    }

    #[rstest]
    #[case::rejected_request(Failure::prepare(Plain::Request("top_k")), true, false, false, true)]
    #[case::unreachable(Failure::from(Exchange::Unreached(Plain::Request("refused"))), true, false, false, true)]
    #[case::provider_400(Failure::from(Exchange::Rejected(upstream(400))), false, true, false, true)]
    #[case::provider_404(Failure::from(Exchange::Rejected(upstream(404))), false, true, false, true)]
    #[case::provider_408(Failure::from(Exchange::Rejected(upstream(408))), false, false, true, false)]
    #[case::provider_429(Failure::from(Exchange::Rejected(upstream(429))), false, false, true, false)]
    #[case::provider_500(Failure::from(Exchange::Rejected(upstream(500))), false, false, true, false)]
    #[case::provider_529(Failure::from(Exchange::Rejected(upstream(529))), false, false, true, false)]
    #[case::timed_out(Failure::from(Exchange::Broken(Plain::Timeout("slow"))), false, false, true, false)]
    #[case::undecodable(Failure::receive(Plain::Request("bad json")), false, false, false, false)]
    #[case::hook_after_answer(Failure::post_call(Plain::Request("rejected")), false, false, false, false)]
    fn the_facts_follow_the_stage_and_the_kind(
        #[case] failure: Failure<Plain>,
        #[case] untouched: bool,
        #[case] rejected: bool,
        #[case] retryable: bool,
        #[case] reroutable: bool,
    ) {
        assert_eq!(failure.provider_untouched(), untouched, "untouched");
        assert_eq!(failure.rejected_by_provider(), rejected, "rejected");
        assert_eq!(failure.is_retryable(), retryable, "retryable");
        assert_eq!(failure.is_reroutable(), reroutable, "reroutable");
        let report = failure.report();
        assert_eq!(
            (
                report.provider_untouched(),
                report.rejected_by_provider(),
                report.is_retryable(),
                report.is_reroutable(),
            ),
            (untouched, rejected, retryable, reroutable),
            "the report answers exactly as the failure does"
        );
    }

    #[rstest]
    fn the_report_carries_the_stage_the_kind_and_the_message() {
        let failure = Failure::from(Exchange::<Plain>::Rejected(upstream(429)));
        assert_eq!(
            failure.report(),
            Report {
                stage: Stage::Upstream,
                kind: Kind::Upstream(upstream(429)),
                message: "upstream request failed with status 429: slow down".into(),
            }
        );

        let failure = Failure::prepare(Plain::Request("top_k"));
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
        let report = Failure::from(Exchange::<Plain>::Rejected(upstream(429))).report();
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
        assert_eq!(
            serde_json::to_value(Kind::Request).unwrap(),
            serde_json::json!({"kind": "request"})
        );
    }

    #[rstest]
    fn map_keeps_the_stage() {
        let mapped = Failure::receive(Plain::Request("x")).map(|error| error.to_string());
        assert_eq!(mapped, Failure::receive("x".to_string()));
    }
}
