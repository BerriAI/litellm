use std::error::Error as StdError;

use litellm_error::Source;

const TOKEN_REJECTED: &str = "token request failed";
const TOKEN_UNREADABLE: &str = "failed to deserialize token from response";
const TOKEN_UNREACHABLE: &str = "HTTP request failed";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CredentialFailure {
    NoDefaultCredentials,
    TokenRejected,
    TokenUnreachable,
    Unloadable,
}

impl CredentialFailure {
    pub fn of(cause: &(dyn StdError + 'static)) -> Self {
        std::iter::successors(Some(cause), |&node| {
            node.downcast_ref::<Source>()
                .map(|source| &**source as &(dyn StdError + 'static))
                .or_else(|| node.source())
        })
        .find_map(Self::at)
        .unwrap_or(Self::Unloadable)
    }

    fn at(node: &(dyn StdError + 'static)) -> Option<Self> {
        if let Some(error) = node.downcast_ref::<gcp_auth::Error>() {
            return match error {
                gcp_auth::Error::NoAuthMethod(..) => Some(Self::NoDefaultCredentials),
                gcp_auth::Error::Str(TOKEN_REJECTED) | gcp_auth::Error::Json(TOKEN_UNREADABLE, _) => {
                    Some(Self::TokenRejected)
                }
                gcp_auth::Error::Other(TOKEN_UNREACHABLE, _) | gcp_auth::Error::Http(..) => {
                    Some(Self::TokenUnreachable)
                }
                _ => None,
            };
        }
        let error = node.downcast_ref::<reqwest::Error>()?;
        Some(if error.is_status() || error.is_decode() {
            Self::TokenRejected
        } else {
            Self::TokenUnreachable
        })
    }
}

#[cfg(test)]
mod tests {
    use std::io;

    use litellm_auth_types::{Error, ErrorDetail};
    use rstest::rstest;

    use super::*;

    fn acquisition(cause: impl StdError + Send + Sync + 'static) -> Error {
        Error::CredentialAcquisition(ErrorDetail::failed("Google credentials", cause))
    }

    fn missing_file() -> io::Error {
        io::Error::new(io::ErrorKind::NotFound, "No such file or directory")
    }

    fn json_error() -> serde_json::Error {
        serde_json::from_str::<serde_json::Value>("{").unwrap_err()
    }

    fn not_found() -> gcp_auth::Error {
        gcp_auth::Error::Str("no project ID in user credentials")
    }

    #[rstest]
    #[case::no_adc(
        gcp_auth::Error::NoAuthMethod(Box::new(not_found()), Box::new(not_found()), Box::new(not_found())),
        CredentialFailure::NoDefaultCredentials
    )]
    #[case::token_endpoint_error_status(gcp_auth::Error::Str(TOKEN_REJECTED), CredentialFailure::TokenRejected)]
    #[case::token_endpoint_unreadable_body(
        gcp_auth::Error::Json(TOKEN_UNREADABLE, json_error()),
        CredentialFailure::TokenRejected
    )]
    #[case::token_endpoint_unreachable(
        gcp_auth::Error::Other(TOKEN_UNREACHABLE, Box::new(io::Error::from(io::ErrorKind::ConnectionRefused))),
        CredentialFailure::TokenUnreachable
    )]
    #[case::invalid_private_key(
        gcp_auth::Error::Str("invalid private key in credentials"),
        CredentialFailure::Unloadable
    )]
    #[case::unreadable_key_file(
        gcp_auth::Error::Io("failed to open application credentials file", missing_file()),
        CredentialFailure::Unloadable
    )]
    fn gcp_auth_failures_are_classified(#[case] cause: gcp_auth::Error, #[case] expected: CredentialFailure) {
        assert_eq!(CredentialFailure::of(&acquisition(cause)), expected);
    }

    #[rstest]
    #[case::missing_file(acquisition(missing_file()))]
    #[case::invalid_json(acquisition(json_error()))]
    #[case::unresolved_project(Error::CredentialAcquisition(ErrorDetail::Unresolved {
        subject: "GCP resource project",
    }))]
    fn failures_outside_the_token_exchange_are_unloadable(#[case] error: Error) {
        assert_eq!(CredentialFailure::of(&error), CredentialFailure::Unloadable);
    }

    async fn token_endpoint_answering(status: u16) -> String {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let app = axum::Router::new().route(
            "/token",
            axum::routing::post(move || async move { axum::http::StatusCode::from_u16(status).unwrap() }),
        );
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        format!("http://{address}/token")
    }

    #[tokio::test]
    async fn an_sdk_token_endpoint_error_status_is_a_rejection() {
        let url = token_endpoint_answering(400).await;
        let rejected = reqwest::Client::new()
            .post(url)
            .send()
            .await
            .unwrap()
            .error_for_status()
            .unwrap_err();
        assert_eq!(CredentialFailure::of(&acquisition(rejected)), CredentialFailure::TokenRejected);
    }

    #[tokio::test]
    async fn an_sdk_token_endpoint_it_cannot_connect_to_is_unreachable() {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let closed = format!("http://{}/token", listener.local_addr().unwrap());
        drop(listener);
        let refused = reqwest::Client::new().post(closed).send().await.unwrap_err();
        assert_eq!(CredentialFailure::of(&acquisition(refused)), CredentialFailure::TokenUnreachable);
    }
}
