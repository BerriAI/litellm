//! The credentials a Vertex AI Messages login builds from what `vertex_credentials` or
//! application default credentials hold: the inputs of Google's Rust SDK builders, chosen as
//! Python's `_credentials_from_json` and google-auth's `default()` choose the credential class,
//! with the checks google-auth makes before it would mint a token.

use std::sync::Arc;

use super::{
    compat_certificate_source,
    environment::GoogleEnvironment,
    explicit_aws::{explicit_aws_credentials, uses_explicit_aws},
    failure::{GoogleLoginFailure, PythonException},
    source::{ExternalAccountInfo, ServiceAccountInfo},
};
use crate::{GcpTokenSource, identity::SdkCredentials};

const NO_CREDENTIAL_SOURCE: &str = "The external account credentials name no credential_source.";
const EXECUTABLES_NOT_ALLOWED: &str = "Executables need to be explicitly allowed (set GOOGLE_EXTERNAL_ACCOUNT_ALLOW_EXECUTABLES to '1') to run.";

/// Builds `credentials` with Google's Rust SDK, a failure raised as `exception`.
pub(crate) fn build(
    credentials: SdkCredentials,
    exception: PythonException,
) -> Result<Arc<dyn GcpTokenSource>, GoogleLoginFailure> {
    credentials
        .build()
        .map_err(|error| GoogleLoginFailure::new(exception, error.to_string()))
}

/// google-auth's `service_account.Credentials.from_service_account_info`, which requires a
/// `client_email`, a `token_uri` and a `private_key`, refused as `exception`.
pub(crate) fn service_account(
    info: &ServiceAccountInfo,
    exception: PythonException,
) -> Result<Arc<dyn GcpTokenSource>, GoogleLoginFailure> {
    let missing: Vec<&str> = [
        ("client_email", &info.client_email),
        ("token_uri", &info.token_uri),
        ("private_key", &info.private_key),
    ]
    .into_iter()
    .filter(|(_, field)| field.is_none())
    .map(|(name, _)| name)
    .collect();
    if !missing.is_empty() {
        return Err(GoogleLoginFailure::new(
            exception,
            format!("The service account key is missing {}.", missing.join(", ")),
        ));
    }
    build(
        SdkCredentials::ServiceAccount(info.sdk.clone()),
        PythonException::Value,
    )
}

/// An external account as Python's `_credentials_from_json` builds it from `vertex_credentials`
/// (`explicit`) or google-auth's `default()` builds it from a file: LiteLLM's explicit AWS login
/// for its own `aws_*` keys, the backwards-compat certificate source, else Google's Rust SDK.
pub(crate) fn external_account(
    info: &ExternalAccountInfo,
    environment: &Arc<dyn GoogleEnvironment>,
    explicit: bool,
) -> Result<Arc<dyn GcpTokenSource>, GoogleLoginFailure> {
    let unusable = match explicit {
        true => PythonException::Malformed,
        false => PythonException::DefaultCredentials,
    };
    if explicit && uses_explicit_aws(info) {
        return build(
            SdkCredentials::Programmatic(explicit_aws_credentials(info, environment.clone())?),
            unusable,
        );
    }
    let Some(source) = &info.credential_source else {
        return Err(GoogleLoginFailure::new(
            match explicit {
                true => PythonException::InvalidValue,
                false => unusable,
            },
            NO_CREDENTIAL_SOURCE,
        ));
    };
    if let Some(certificate) = &source.certificate {
        return compat_certificate_source::load(info, certificate, environment.clone());
    }
    if source.executable.is_some()
        && environment
            .os_env("GOOGLE_EXTERNAL_ACCOUNT_ALLOW_EXECUTABLES")
            .as_deref()
            != Some("1")
    {
        return Err(GoogleLoginFailure::new(
            PythonException::Malformed,
            EXECUTABLES_NOT_ALLOWED,
        ));
    }
    build(SdkCredentials::ExternalAccount(info.sdk.clone()), unusable)
}
