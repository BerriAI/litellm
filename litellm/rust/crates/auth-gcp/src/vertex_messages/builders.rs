//! The credentials Google's Rust SDK builds from a credentials file, google-auth's checks of a
//! service account key, and the failure google-auth raises when minting a token fails.

use std::{error::Error as StdError, sync::Arc};

use google_cloud_auth::credentials::{AccessTokenCredentials, external_account, impersonated};
use serde_json::Value;

use super::{
    credentials::Loaded,
    external_account::ExternalAccountInfo,
    failure::{GoogleLoginFailure, PythonException},
    service_account::{ServiceAccountInfo, service_account_grant},
    source::{ImpersonatedInfo, SdkJson},
};
use crate::{
    constants::CLOUD_PLATFORM_SCOPE,
    identity::{GcpTokenSource, SdkTokenSource},
};

pub(crate) type TokenSource = Arc<dyn GcpTokenSource>;

const NO_CREDENTIAL_SOURCE: &str =
    "A valid credential source or a subject token supplier must be provided.";

fn value_failure(error: impl std::fmt::Display) -> GoogleLoginFailure {
    GoogleLoginFailure::new(PythonException::Value, error.to_string())
}

/// google-auth keeps a quota project of any JSON type, and the Vertex login only takes the token.
fn without_quota_project(info: &SdkJson) -> Value {
    Value::Object(
        info.0
            .iter()
            .filter(|(field, _)| field.as_str() != "quota_project_id")
            .map(|(field, value)| (field.clone(), value.clone()))
            .collect(),
    )
}

fn sdk_source(credentials: AccessTokenCredentials) -> TokenSource {
    Arc::new(SdkTokenSource {
        credentials,
        project_id: None,
    })
}

/// google-auth's `service_account.Credentials.from_service_account_info`.
pub(crate) fn service_account(info: &ServiceAccountInfo) -> Result<Loaded, GoogleLoginFailure> {
    let missing: Vec<&str> = [
        ("client_email", &info.client_email),
        ("token_uri", &info.token_uri),
    ]
    .into_iter()
    .filter(|(_, field)| field.is_none())
    .map(|(name, _)| name)
    .collect();
    if !missing.is_empty() {
        let missing = missing.join(", ");
        return Err(GoogleLoginFailure::new(
            PythonException::Malformed,
            format!(
                "Service account info was not in the expected format, missing fields {missing}."
            ),
        ));
    }
    if info.private_key.is_none() {
        return Err(GoogleLoginFailure::new(
            PythonException::Malformed,
            "The private_key field was not found in the service account info.",
        ));
    }
    service_account_grant(info)
}

/// google-auth's identity pool credentials with a certificate source, which mint through mutual
/// TLS in Google's Rust SDK.
pub(crate) fn external_account(
    info: &ExternalAccountInfo,
) -> Result<TokenSource, GoogleLoginFailure> {
    if info.credential_source.is_none() {
        return Err(GoogleLoginFailure::new(
            PythonException::InvalidValue,
            NO_CREDENTIAL_SOURCE,
        ));
    }
    external_account::Builder::new(without_quota_project(&info.sdk))
        .with_scopes([CLOUD_PLATFORM_SCOPE])
        .build_access_token_credentials()
        .map(sdk_source)
        .map_err(|error| GoogleLoginFailure::new(PythonException::InvalidValue, error.to_string()))
}

/// google-auth's `impersonated_credentials.Credentials.from_impersonated_service_account_info`.
pub(crate) fn impersonated_service_account(
    info: &ImpersonatedInfo,
) -> Result<TokenSource, GoogleLoginFailure> {
    impersonated::Builder::new(without_quota_project(&info.sdk))
        .with_scopes([CLOUD_PLATFORM_SCOPE])
        .build_access_token_credentials()
        .map(sdk_source)
        .map_err(value_failure)
}

fn is_transport(error: &(dyn StdError + 'static)) -> bool {
    litellm_error::find::<reqwest::Error>(error)
        .is_some_and(|error| error.is_connect() || error.is_timeout())
        || litellm_error::find::<std::io::Error>(error).is_some()
}

fn innermost(error: &(dyn StdError + 'static)) -> String {
    std::iter::successors(Some(error), |&cause| cause.source())
        .last()
        .map_or_else(String::new, ToString::to_string)
}

/// The exception google-auth's `refresh` raises when minting a token fails: a `TransportError`
/// when the token endpoint could not be reached, else a `RefreshError`.
pub(crate) fn token_failure(error: &(dyn StdError + 'static)) -> GoogleLoginFailure {
    if let Some(failure) = litellm_error::find::<GoogleLoginFailure>(error) {
        return failure.clone();
    }
    let exception = if is_transport(error) {
        PythonException::Transport
    } else {
        PythonException::Refresh
    };
    GoogleLoginFailure::new(exception, innermost(error))
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use serde_json::{Value, json};

    use super::*;

    fn info(value: Value) -> ServiceAccountInfo {
        serde_json::from_value(value).unwrap()
    }

    fn failure<T>(result: Result<T, GoogleLoginFailure>) -> GoogleLoginFailure {
        match result {
            Ok(_) => panic!("the credentials should not build"),
            Err(failure) => failure,
        }
    }

    #[rstest]
    #[case::impersonated_type(
        json!({"type": "impersonated_service_account", "client_email": "a@b"}),
        "Service account info was not in the expected format, missing fields token_uri."
    )]
    #[case::unknown_type(
        json!({"type": "weird"}),
        "Service account info was not in the expected format, missing fields client_email, token_uri."
    )]
    #[case::no_private_key(
        json!({"client_email": "a@b", "token_uri": "https://oauth2.googleapis.com/token"}),
        "The private_key field was not found in the service account info."
    )]
    fn service_account_info_is_checked_as_google_auth_checks_it(
        #[case] configured: Value,
        #[case] expected: &str,
    ) {
        let failure = failure(service_account(&info(configured)));
        assert_eq!(failure.exception, PythonException::Malformed);
        assert_eq!(failure.message, expected);
    }

    #[rstest]
    fn an_unparseable_private_key_is_a_value_error() {
        let failure = failure(service_account(&info(json!({
            "client_email": "a@b", "token_uri": "https://oauth2.googleapis.com/token", "private_key": "nope"
        }))));
        assert_eq!(failure.exception, PythonException::Value);
    }

    #[rstest]
    fn an_external_account_without_a_source_is_an_invalid_value() {
        let failure = failure(external_account(&ExternalAccountInfo::default()));
        assert_eq!(failure.exception, PythonException::InvalidValue);
        assert_eq!(failure.message, NO_CREDENTIAL_SOURCE);
    }

    #[rstest]
    fn a_login_failure_in_the_chain_is_kept() {
        let raised = GoogleLoginFailure::new(PythonException::Value, "aws_region_name is required");
        let wrapped = litellm_auth_types::Error::from(raised.clone());
        assert_eq!(token_failure(&wrapped), raised);
    }

    #[rstest]
    fn an_unreachable_endpoint_is_a_transport_error() {
        let wrapped = litellm_auth_types::Error::provider(std::io::Error::from(
            std::io::ErrorKind::ConnectionRefused,
        ));
        assert_eq!(
            token_failure(&wrapped).exception,
            PythonException::Transport
        );
    }

    #[rstest]
    fn a_rejected_refresh_is_a_refresh_error() {
        let wrapped = litellm_auth_types::Error::CredentialAcquisition(
            litellm_auth_types::ErrorDetail::Unresolved {
                subject: "Google access token",
            },
        );
        assert_eq!(token_failure(&wrapped).exception, PythonException::Refresh);
    }
}
