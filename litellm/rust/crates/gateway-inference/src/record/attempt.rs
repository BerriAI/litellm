//! What one attempt saw of its provider, and the deployment that served the call. Each attempt
//! runs in its own [`AttemptRecord`], which [`Recorder`] fills from the core's interceptor hooks,
//! and only the served attempt's reaches the call's record.

use std::time::{Duration, Instant};

use litellm_host::interceptors::{
    ExecutionFacts, Interceptors, RawResponse, RequestContext, ResultSource, RetriedRequest,
    WireRequest,
};

use crate::Error;

/// What one attempt saw of its provider: where it called, when its request went out and its
/// response head came back, the headers that head carried and where its result came from.
/// [`Recorder`] fills it; readers get it through the call's record, which only lends it out.
/// `Debug` hides the URL and the headers, which can carry credentials.
#[derive(Clone, Default, PartialEq, Eq, veil::Redact)]
pub struct AttemptRecord {
    /// The wire URL the attempt called.
    #[redact(with = "[REDACTED]")]
    pub url: Option<String>,
    /// When the provider request was sent, a retried one by its resend.
    pub sent: Option<Instant>,
    /// When the provider's response head arrived, for a stream when it opened.
    pub head: Option<Instant>,
    /// How long the provider took from the request sent to its response head.
    pub wait: Option<Duration>,
    /// The headers the provider's response head carried.
    #[redact(with = "[REDACTED]")]
    pub provider_headers: Vec<(String, String)>,
    /// Where the attempt's result came from.
    pub source: Option<ResultSource>,
}

impl AttemptRecord {
    /// The key of the cached response the attempt was served from.
    pub fn cache_key(&self) -> Option<&str> {
        match self.source.as_ref()? {
            ResultSource::Cache { key } => Some(key),
            ResultSource::Provider | ResultSource::Host => None,
        }
    }

    fn record_sent(&mut self, url: Option<String>) {
        if url.is_some() {
            self.url = url;
        }
        self.sent = Some(Instant::now());
    }

    fn record_response(&mut self, headers: Option<Vec<(String, String)>>) {
        let head = Instant::now();
        self.head = Some(head);
        self.wait = self.sent.map(|sent| head.saturating_duration_since(sent));
        if let Some(headers) = headers {
            self.provider_headers = headers;
        }
    }

    /// The response head of a provider failure `error` carries, for an attempt whose core
    /// reported no response head.
    pub(super) fn record_failure(&mut self, error: &Error) {
        if self.head.is_some() {
            return;
        }
        if let Error::Route(route) = error
            && route.upstream_response().is_some()
        {
            self.record_response(Some(route.upstream_headers().to_vec()));
        }
    }

    /// An attempt served from the cached response `key`.
    #[cfg(test)]
    pub(crate) fn cached(key: &str) -> Self {
        Self {
            source: Some(ResultSource::Cache {
                key: key.to_owned(),
            }),
            ..Self::default()
        }
    }
}

/// The deployment that served a call, as the router registry and `routing::timeout` name it.
/// `Debug` hides the `api_base`, which can carry credentials.
#[derive(Clone, PartialEq, Eq, veil::Redact)]
pub struct ServedDeployment {
    /// The provider's model name the deployment calls.
    pub model: String,
    #[redact(with = "[REDACTED]")]
    pub api_base: Option<String>,
    pub timeout: Option<Duration>,
}

/// The core's interceptor hooks as they reach the attempt in scope. It holds nothing itself, so
/// a retried or replaced attempt leaves nothing behind.
#[derive(Clone, Copy, Debug, Default)]
pub struct Recorder;

impl<E> Interceptors<E> for Recorder {
    async fn before_provider_request(
        &self,
        wire: WireRequest,
        _: RequestContext,
    ) -> Result<WireRequest, E> {
        super::note(|record| record.record_sent(Some(wire.url.clone())));
        Ok(wire)
    }

    async fn provider_request_retried(&self, _: RetriedRequest) -> Result<(), E> {
        super::note(|record| record.record_sent(None));
        Ok(())
    }

    async fn after_provider_response(&self, raw: RawResponse) -> Result<(), E> {
        super::note(|record| record.record_response(raw.head.map(|head| head.headers)));
        Ok(())
    }

    async fn result_ready(&self, facts: ExecutionFacts) -> Result<(), E> {
        super::note(|record| record.source = Some(facts.source));
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use litellm_host::interceptors::{Cost, ProviderIdentity, ResponseHead};
    use rstest::rstest;
    use serde_json::json;

    use super::*;

    type Seen = (Option<String>, bool, Vec<(String, String)>, Option<String>);

    fn wire() -> WireRequest {
        WireRequest {
            url: "https://provider.test/v1/messages".to_owned(),
            headers: Vec::new(),
            body: json!({}),
        }
    }

    fn context() -> RequestContext {
        RequestContext {
            model: "m".to_owned(),
            custom_llm_provider: "anthropic".to_owned(),
            optional_params: json!({}),
            secret_fields: Vec::new(),
            api_key: None,
            host_transport: false,
        }
    }

    fn head() -> RawResponse {
        RawResponse::new(String::new()).with_head(ResponseHead {
            status: 200,
            headers: vec![("request-id".to_owned(), "req_1".to_owned())],
            sent_body: bytes::Bytes::new(),
            streamed: true,
        })
    }

    fn ready(source: ResultSource) -> ExecutionFacts {
        ExecutionFacts {
            provider: ProviderIdentity {
                model: "m".to_owned(),
                provider: "anthropic".to_owned(),
            },
            source,
            cost: Cost::Deferred,
        }
    }

    /// What `Recorder` left in the attempt in scope once the core told it of a request `sends`, a
    /// response head `responds` and a result from `source`.
    async fn seen(sends: bool, responds: bool, source: ResultSource) -> Seen {
        let (_, record) = super::super::attempt(async {
            let recorder = &Recorder;
            if sends {
                Interceptors::<()>::before_provider_request(recorder, wire(), context())
                    .await
                    .unwrap();
            }
            if responds {
                Interceptors::<()>::after_provider_response(recorder, head())
                    .await
                    .unwrap();
            }
            Interceptors::<()>::result_ready(recorder, ready(source))
                .await
                .map_err(|()| Error::Internal)
        })
        .await;
        (
            record.url.clone(),
            record.wait.is_some() && record.head >= record.sent,
            record.provider_headers.clone(),
            record.cache_key().map(str::to_owned),
        )
    }

    #[rstest]
    #[case::a_provider_response(
        true,
        true,
        ResultSource::Provider,
        (
            Some("https://provider.test/v1/messages".to_owned()),
            true,
            vec![("request-id".to_owned(), "req_1".to_owned())],
            None,
        )
    )]
    #[case::a_cache_hit(
        false,
        false,
        ResultSource::Cache { key: "k1".to_owned() },
        (None, false, Vec::new(), Some("k1".to_owned()))
    )]
    #[tokio::test]
    async fn the_recorder_writes_what_the_core_tells_it_into_the_attempt_in_scope(
        #[case] sends: bool,
        #[case] responds: bool,
        #[case] source: ResultSource,
        #[case] expected: Seen,
    ) {
        assert_eq!(seen(sends, responds, source).await, expected);
    }

    /// What a failed attempt that sent its request left in its record, inside a call.
    async fn failed(error: Error) -> (Vec<(String, String)>, bool) {
        let (_, record) = super::super::scope(super::super::CallRecord::new(), async {
            super::super::attempt(async {
                Interceptors::<()>::before_provider_request(&Recorder, wire(), context())
                    .await
                    .map_err(|()| Error::Internal)?;
                Err::<(), _>(error)
            })
            .await
        })
        .await;
        (record.provider_headers, record.wait.is_some())
    }

    #[rstest]
    #[case::a_provider_failure(
        Error::Route(
            litellm_http::Error::upstream_with_headers(
                529,
                "overloaded".to_owned(),
                vec![("request-id".to_owned(), "req_1".to_owned())],
            )
            .into(),
        ),
        (vec![("request-id".to_owned(), "req_1".to_owned())], true)
    )]
    #[case::no_provider_answer(Error::Internal, (Vec::new(), false))]
    #[tokio::test]
    async fn a_failed_attempt_keeps_the_response_head_its_error_carries(
        #[case] error: Error,
        #[case] expected: (Vec<(String, String)>, bool),
    ) {
        assert_eq!(failed(error).await, expected);
    }

    #[rstest]
    #[case::an_attempt(format!(
        "{:?}",
        AttemptRecord {
            url: Some("https://user:credential-value@provider.test/v1".to_owned()),
            provider_headers: vec![("x-token".to_owned(), "credential-value".to_owned())],
            ..AttemptRecord::default()
        }
    ))]
    #[case::a_served_deployment(format!(
        "{:?}",
        ServedDeployment {
            model: "m".to_owned(),
            api_base: Some("https://user:credential-value@provider.test".to_owned()),
            timeout: None,
        }
    ))]
    fn debug_hides_what_can_carry_credentials(#[case] debug: String) {
        assert!(!debug.contains("credential-value") && debug.contains("[REDACTED]"));
    }
}
