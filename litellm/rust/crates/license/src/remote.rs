use std::time::Duration;

use backon::{ConstantBuilder, Retryable};
use litellm_http::Client;
use serde_json::Value;

use crate::Error;

pub const LICENSE_SERVER: &str = "https://license.litellm.ai";

/// Asks the license server, retrying a non-2xx reply right away up to three attempts in all. A transport failure is
/// final
pub(crate) async fn verify_remote(
    client: &Client,
    server: &str,
    license: &str,
    timeout: Duration,
) -> Result<bool, Error> {
    let url = format!("{server}/verify_license/{license}");
    let send = || async {
        let response = client
            .get(&url)
            .timeout(timeout)
            .send()
            .await
            .map_err(|error| Error::Server(error.into()))?;
        let status = response.status();
        if !status.is_success() {
            return Err(Error::Status(status.as_u16()));
        }
        response
            .bytes()
            .await
            .map_err(|error| Error::Server(error.into()))
    };
    let body = send
        .retry(
            ConstantBuilder::new()
                .with_delay(Duration::ZERO)
                .with_max_times(2),
        )
        .when(|error| matches!(error, Error::Status(_)))
        .await?;
    let reply: Value = serde_json::from_slice(&body).map_err(|error| Error::Reply(Some(error)))?;
    reply
        .get("verify")
        .and_then(Value::as_bool)
        .ok_or(Error::Reply(None))
}

#[cfg(test)]
mod tests {
    use litellm_error::Report;
    use rstest::rstest;
    use serde_json::json;
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    use super::*;

    const LICENSE: &str = "test-only-license-marker";
    const TIMEOUT: Duration = Duration::from_secs(5);

    async fn verify(server: &MockServer) -> Result<bool, Error> {
        verify_remote(&Client::for_test(), &server.uri(), LICENSE, TIMEOUT).await
    }

    async fn reply_with(template: ResponseTemplate, expected_requests: u64) -> MockServer {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path(format!("/verify_license/{LICENSE}")))
            .respond_with(template)
            .expect(expected_requests)
            .mount(&server)
            .await;
        server
    }

    #[rstest]
    #[case::premium(json!({"verify": true}), true)]
    #[case::not_premium(json!({"verify": false}), false)]
    #[tokio::test]
    async fn returns_the_server_verdict(#[case] reply: Value, #[case] premium: bool) {
        let server = reply_with(ResponseTemplate::new(200).set_body_json(reply), 1).await;

        assert_eq!(verify(&server).await.unwrap(), premium);
    }

    #[rstest]
    #[case::array(ResponseTemplate::new(200).set_body_json(json!(["verify", true])))]
    #[case::single_element_array(ResponseTemplate::new(200).set_body_json(json!([true])))]
    #[case::string(ResponseTemplate::new(200).set_body_json(json!({"verify": "true"})))]
    #[case::other_field(ResponseTemplate::new(200).set_body_json(json!({"verified": true})))]
    #[case::not_json(ResponseTemplate::new(200).set_body_string("true"))]
    #[tokio::test]
    async fn rejects_a_reply_without_a_boolean_verify(#[case] reply: ResponseTemplate) {
        let server = reply_with(reply, 1).await;

        assert!(matches!(verify(&server).await, Err(Error::Reply(_))));
    }

    #[rstest]
    #[case::not_found(404)]
    #[case::unavailable(503)]
    #[case::redirect(302)]
    #[tokio::test]
    async fn gives_up_after_three_failed_statuses(#[case] status: u16) {
        let server = reply_with(ResponseTemplate::new(status), 3).await;

        assert!(matches!(verify(&server).await, Err(Error::Status(s)) if s == status));
    }

    #[rstest]
    #[tokio::test]
    async fn retries_until_a_success() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .respond_with(ResponseTemplate::new(500))
            .up_to_n_times(2)
            .expect(2)
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({"verify": true})))
            .expect(1)
            .mount(&server)
            .await;

        assert!(verify(&server).await.unwrap());
    }

    #[rstest]
    #[tokio::test]
    async fn does_not_retry_a_transport_failure_or_report_the_license() {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let closed = format!("http://{}", listener.local_addr().unwrap());
        drop(listener);

        let error = verify_remote(&Client::for_test(), &closed, LICENSE, TIMEOUT)
            .await
            .unwrap_err();

        assert!(matches!(error, Error::Server(_)));
        assert!(!Report::new(&error).to_string().contains(LICENSE));
        assert!(!format!("{error:?}").contains(LICENSE));
    }
}
