//! The shared steps of one provider call. Routes reach a [`Stage`] only through these, so no
//! route decides on its own where a failure happened or what a provider's answer means.

use bytes::BytesMut;
pub use litellm_host::failure::{
    Classify, Exchange, Failure, Kind, Report, Stage, UpstreamResponse, post_call, prepare,
    receive,
};
use litellm_http::{outbound::OutboundRequest, transport};

/// How much of a provider's error body is kept when a route sets no limit of its own.
pub const UPSTREAM_ERROR_BODY_MAX_BYTES: usize = 64 * 1024;

/// Sends the request and hands back a success response. A connection that never opened is
/// [`Stage::Send`]; a provider answer that is not a success is [`Stage::Upstream`]; a request
/// that timed out or broke mid-flight may have reached the provider and is [`Stage::Receive`].
#[tracing::instrument(
    name = "litellm.provider.send",
    level = "debug",
    skip_all,
    fields(status)
)]
pub async fn send<E>(
    client: &litellm_http::Client,
    request: OutboundRequest,
) -> Result<reqwest::Response, Failure<E>>
where
    E: From<transport::Error> + From<UpstreamResponse>,
{
    let response = match request.send(client).await {
        Ok(response) => response,
        Err(error) => {
            let error = transport::Error::from_reqwest_before_dispatch(error);
            let exchange = match error {
                transport::Error::Connect(_) => Exchange::Unreached(E::from(error)),
                transport::Error::Network(_) | transport::Error::Timeout(_) => {
                    Exchange::Broken(E::from(error))
                }
            };
            return Err(exchange.into());
        }
    };
    tracing::Span::current().record("status", response.status().as_u16());
    if response.status().is_success() {
        return Ok(response);
    }
    match upstream_response(response, UPSTREAM_ERROR_BODY_MAX_BYTES).await {
        Ok(upstream) => Err(Exchange::Rejected(upstream).into()),
        Err(error) => Err(Exchange::Broken(E::from(error)).into()),
    }
}

/// Reads a non-success answer, keeping at most `limit` bytes of its body so an oversized
/// error page is never drained. Within the limit the body is the provider's, word for word.
pub async fn upstream_response(
    mut response: reqwest::Response,
    limit: usize,
) -> Result<UpstreamResponse, transport::Error> {
    let status = response.status().as_u16();
    let url = Some(response.url().to_string());
    let headers = response
        .headers()
        .iter()
        .filter_map(|(name, value)| Some((name.to_string(), value.to_str().ok()?.to_owned())))
        .collect();
    let mut bytes = BytesMut::new();
    while bytes.len() < limit
        && let Some(chunk) = response.chunk().await?
    {
        let remaining = limit - bytes.len();
        bytes.extend_from_slice(&chunk[..chunk.len().min(remaining)]);
    }
    let body = String::from_utf8_lossy(&bytes).into_owned();
    tracing::debug!(status, body = %body, "provider error body");
    Ok(UpstreamResponse {
        status,
        headers,
        body,
        url,
    })
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use rstest::rstest;
    use wiremock::{Mock, MockServer, ResponseTemplate, matchers::any};

    use super::*;

    #[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
    enum TestError {
        #[error(transparent)]
        Transport(#[from] transport::Error),
        #[error(transparent)]
        Upstream(#[from] UpstreamResponse),
    }

    fn request(url: String, timeout: Option<Duration>) -> OutboundRequest {
        OutboundRequest::json(url, Vec::new(), &serde_json::json!({}), timeout).unwrap()
    }

    #[rstest]
    #[tokio::test]
    async fn a_success_passes_through_untouched() {
        let server = MockServer::start().await;
        Mock::given(any())
            .respond_with(ResponseTemplate::new(200).set_body_string("ok"))
            .mount(&server)
            .await;
        let response = send::<TestError>(
            &litellm_http::Client::plain_for_test(),
            request(server.uri(), None),
        )
        .await
        .unwrap();
        assert_eq!(response.text().await.unwrap(), "ok");
    }

    #[rstest]
    #[tokio::test]
    async fn a_provider_answer_is_the_upstream_stage_with_status_headers_body_and_url() {
        let server = MockServer::start().await;
        Mock::given(any())
            .respond_with(
                ResponseTemplate::new(429)
                    .insert_header("retry-after", "7")
                    .set_body_string("slow down"),
            )
            .mount(&server)
            .await;
        let failure = send::<TestError>(
            &litellm_http::Client::plain_for_test(),
            request(server.uri(), None),
        )
        .await
        .unwrap_err();
        assert_eq!(failure.stage, Stage::Upstream);
        let TestError::Upstream(upstream) = failure.error else {
            panic!("{:?}", failure.error);
        };
        assert_eq!(upstream.status, 429);
        assert_eq!(upstream.body, "slow down");
        assert_eq!(
            upstream.url.as_deref(),
            Some(format!("{}/", server.uri()).as_str())
        );
        assert!(
            upstream
                .headers
                .contains(&("retry-after".into(), "7".into()))
        );
    }

    #[rstest]
    #[tokio::test]
    async fn an_unreachable_provider_is_the_send_stage() {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        drop(listener);
        let failure = send::<TestError>(
            &litellm_http::Client::plain_for_test(),
            request(format!("http://{address}"), None),
        )
        .await
        .unwrap_err();
        assert_eq!(failure.stage, Stage::Send);
        assert!(matches!(
            failure.error,
            TestError::Transport(transport::Error::Connect(_))
        ));
    }

    #[rstest]
    #[tokio::test]
    async fn a_timeout_may_have_reached_the_provider_and_is_the_receive_stage() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            let _connection = listener.accept().await.unwrap();
            tokio::time::sleep(Duration::from_secs(2)).await;
        });
        let failure = send::<TestError>(
            &litellm_http::Client::plain_for_test(),
            request(format!("http://{address}"), Some(Duration::from_millis(20))),
        )
        .await
        .unwrap_err();
        server.abort();
        assert_eq!(failure.stage, Stage::Receive);
        assert!(matches!(
            failure.error,
            TestError::Transport(transport::Error::Timeout(_))
        ));
    }

    #[rstest]
    #[tokio::test]
    async fn an_oversized_error_body_is_bounded_without_draining() {
        let server = MockServer::start().await;
        Mock::given(any())
            .respond_with(ResponseTemplate::new(500).set_body_string("x".repeat(10_000)))
            .mount(&server)
            .await;
        let response = litellm_http::Client::plain_for_test()
            .get(server.uri())
            .send()
            .await
            .unwrap();
        let upstream = upstream_response(response, 1_000).await.unwrap();
        assert_eq!(upstream.status, 500);
        assert_eq!(upstream.body, "x".repeat(1_000));
    }
}
