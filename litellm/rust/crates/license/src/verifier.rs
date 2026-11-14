use std::time::Duration;

use litellm_error::Report;
use litellm_http::Client;

use crate::remote::verify_remote;
use crate::{Entitlements, Error, PublicKey};

/// Checks a license key the way the proxy does: a signature from the trusted key first, then the license server
pub struct Verifier {
    public_key: PublicKey,
    server: String,
    timeout: Duration,
}

impl Verifier {
    pub fn new(public_key: PublicKey, server: String, timeout: Duration) -> Self {
        Self {
            public_key,
            server,
            timeout,
        }
    }

    /// `client` is built only for a license the key did not sign, so a signed license never touches the network
    pub async fn entitlements(
        &self,
        license: &str,
        client: impl FnOnce() -> Result<Client, litellm_http::Error>,
    ) -> Entitlements {
        let rejection = match self.public_key.verify_now(license) {
            Ok(signed) => return Entitlements::Signed(signed),
            Err(rejection) => rejection,
        };
        tracing::debug!("license not verified locally: {}", Report::new(&rejection));
        let verdict = match client() {
            Ok(client) => verify_remote(&client, &self.server, license, self.timeout).await,
            Err(error) => Err(Error::Server(error)),
        };
        match verdict {
            Ok(true) => Entitlements::ServerVerified,
            Ok(false) => Entitlements::Community,
            Err(error) => {
                tracing::error!(
                    "license not verified by the license server: {}",
                    Report::new(&error)
                );
                Entitlements::Community
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use serde_json::{Value, json};
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    use super::*;
    use crate::test_support::{license, public_key};

    const UNSIGNED: &str = "test-only-license-marker";

    fn verifier(server: &MockServer) -> Verifier {
        Verifier::new(public_key(), server.uri(), Duration::from_secs(5))
    }

    async fn license_server(reply: ResponseTemplate, expected_requests: u64) -> MockServer {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .respond_with(reply)
            .expect(expected_requests)
            .mount(&server)
            .await;
        server
    }

    #[rstest]
    #[tokio::test]
    async fn a_signed_license_never_builds_a_client() {
        let server = license_server(ResponseTemplate::new(200), 0).await;

        let entitlements = verifier(&server)
            .entitlements(&license("2999-01-01"), || {
                panic!("a signed license must not reach the license server")
            })
            .await;

        assert!(matches!(entitlements, Entitlements::Signed(_)));
    }

    #[rstest]
    #[case::vouched(json!({"verify": true}), Entitlements::ServerVerified)]
    #[case::refused(json!({"verify": false}), Entitlements::Community)]
    #[tokio::test]
    async fn an_unsigned_license_takes_the_server_verdict(
        #[case] reply: Value,
        #[case] expected: Entitlements,
    ) {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path(format!("/verify_license/{UNSIGNED}")))
            .respond_with(ResponseTemplate::new(200).set_body_json(reply))
            .expect(1)
            .mount(&server)
            .await;

        let entitlements = verifier(&server)
            .entitlements(UNSIGNED, || Ok(Client::for_test()))
            .await;

        assert_eq!(entitlements, expected);
    }

    #[rstest]
    #[tokio::test]
    async fn an_expired_signed_license_falls_back_to_the_server() {
        let server = license_server(
            ResponseTemplate::new(200).set_body_json(json!({"verify": true})),
            1,
        )
        .await;

        let entitlements = verifier(&server)
            .entitlements(&license("2000-01-01"), || Ok(Client::for_test()))
            .await;

        assert_eq!(entitlements, Entitlements::ServerVerified);
    }

    #[rstest]
    #[tokio::test]
    async fn a_failing_license_server_leaves_community() {
        let server = license_server(ResponseTemplate::new(503), 3).await;

        let entitlements = verifier(&server)
            .entitlements(UNSIGNED, || Ok(Client::for_test()))
            .await;

        assert_eq!(entitlements, Entitlements::Community);
    }
}
