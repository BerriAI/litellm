//! The credentials a Google login builds from a credentials file, inline credentials or
//! application default credentials: the inputs of Google's Rust SDK builders, chosen as
//! Python's `_credentials_from_json` and google-auth's `default()` choose the credential class,
//! with the checks google-auth makes before it would mint a token.

use std::sync::Arc;

use google_cloud_auth::credentials::{
    external_account::ProgrammaticBuilder, subject_token::SubjectTokenProvider,
};
use rsa::{
    RsaPrivateKey,
    pkcs1::DecodeRsaPrivateKey,
    pkcs8::{EncodePrivateKey, LineEnding},
};
use serde_json::Value;

use super::{
    bounded_file_source,
    credentials::{ExternalAccountInfo, ServiceAccountInfo, target_principal},
    environment::GoogleEnvironment,
    failure::{GoogleLoginFailure, PythonException, UNLOADABLE_CREDENTIALS},
    legacy,
};
use crate::{
    GcpTokenSource,
    identity::{SdkCredentials, SdkJson},
};

const NO_CREDENTIAL_SOURCE: &str = "The external account credentials name no credential_source.";
const NO_PRINCIPAL: &str = "The service account impersonation URL names no service account.";
pub(crate) const AWS_SOURCE_IN_FIPS: &str = "The FIPS build cannot use an AWS credential_source through Google's Rust SDK, which signs the AWS request outside the FIPS module. Put the credentials in vertex_credentials with LiteLLM's aws_region_name and other aws_* keys so LiteLLM signs the request.";

/// Builds `credentials` with Google's Rust SDK, a failure raised as `exception`.
pub(crate) fn build(
    credentials: SdkCredentials,
    exception: PythonException,
) -> Result<Arc<dyn GcpTokenSource>, GoogleLoginFailure> {
    credentials
        .build()
        .map_err(|_| GoogleLoginFailure::new(exception, UNLOADABLE_CREDENTIALS))
}

/// The external account `info` names, built as Google's Rust SDK builds it from the file but
/// with `provider` supplying its subject token, refused as `unusable` without the audience and
/// subject token type the SDK requires.
pub(crate) fn programmatic<P>(
    info: &ExternalAccountInfo,
    provider: P,
    unusable: PythonException,
) -> Result<ProgrammaticBuilder, GoogleLoginFailure>
where
    P: SubjectTokenProvider<Error = GoogleLoginFailure> + 'static,
{
    let (Some(audience), Some(subject_token_type)) = (&info.audience, &info.subject_token_type)
    else {
        return Err(GoogleLoginFailure::new(unusable, UNLOADABLE_CREDENTIALS));
    };
    let builder = ProgrammaticBuilder::new(Arc::new(provider))
        .with_audience(audience)
        .with_subject_token_type(subject_token_type);
    let builder = info
        .token_url
        .iter()
        .fold(builder, |builder, url| builder.with_token_url(url));
    let builder = info
        .client_id
        .iter()
        .fold(builder, |builder, id| builder.with_client_id(id));
    let builder = info.client_secret.iter().fold(builder, |builder, secret| {
        builder.with_client_secret(secret)
    });
    let builder = info
        .workforce_pool_user_project
        .iter()
        .fold(builder, |builder, project| {
            builder.with_workforce_pool_user_project(project)
        });
    Ok(info
        .universe_domain
        .iter()
        .fold(builder, |builder, domain| {
            builder.with_universe_domain(domain)
        }))
}

/// The STS exchange `builder` makes, then the service account `info` names impersonated: by
/// Google's Rust SDK on its own host, else at the named URL through [`legacy::impersonated_at`].
pub(crate) fn exchanged(
    info: &ExternalAccountInfo,
    builder: ProgrammaticBuilder,
    environment: &Arc<dyn GoogleEnvironment>,
    unusable: PythonException,
) -> Result<Arc<dyn GcpTokenSource>, GoogleLoginFailure> {
    let Some(url) = info.service_account_impersonation_url.as_deref() else {
        return build(SdkCredentials::Programmatic(builder), unusable);
    };
    let principal = target_principal(url)
        .ok_or_else(|| GoogleLoginFailure::new(PythonException::Refresh, NO_PRINCIPAL))?;
    match legacy::sdk_impersonates_at(url) {
        true => build(
            SdkCredentials::Programmatic(builder.with_target_principal(principal)),
            unusable,
        ),
        false => Ok(legacy::impersonated_at(
            build(SdkCredentials::Programmatic(builder), unusable)?,
            url,
            info.service_account_impersonation
                .as_ref()
                .and_then(|options| options.token_lifetime_seconds),
            environment.clone(),
        )),
    }
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
        SdkCredentials::ServiceAccount(with_pkcs8_private_key(info)),
        PythonException::Value,
    )
}

/// The key with a PKCS#1 `private_key` re-encoded as the PKCS#8 Google's Rust SDK reads, since
/// google-auth signs with either.
fn with_pkcs8_private_key(info: &ServiceAccountInfo) -> SdkJson {
    let pkcs8 = info
        .private_key
        .as_deref()
        .and_then(|key| RsaPrivateKey::from_pkcs1_pem(key).ok())
        .and_then(|key| key.to_pkcs8_pem(LineEnding::LF).ok());
    match pkcs8 {
        Some(pem) => SdkJson(
            info.sdk
                .0
                .iter()
                .map(|(field, value)| match field.as_str() {
                    "private_key" => (field.clone(), Value::String(pem.to_string())),
                    _ => (field.clone(), value.clone()),
                })
                .collect(),
        ),
        None => info.sdk.clone(),
    }
}

/// An external account as Python's `_credentials_from_json` builds it from `vertex_credentials`
/// (`explicit`) or google-auth's `default()` builds it from a file: the backwards-compat
/// certificate and executable sources, a file source read bounded, else Google's Rust SDK.
/// LiteLLM's explicit AWS login is [`explicit_aws`].
pub(crate) fn external_account(
    info: &ExternalAccountInfo,
    environment: &Arc<dyn GoogleEnvironment>,
    explicit: bool,
) -> Result<Arc<dyn GcpTokenSource>, GoogleLoginFailure> {
    let unusable = match explicit {
        true => PythonException::Malformed,
        false => PythonException::DefaultCredentials,
    };
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
        return legacy::certificate_source(info, certificate, environment.clone());
    }
    if let (None, Some(executable)) = (&source.environment_id, &source.executable) {
        return legacy::executable_source(info, executable, environment, explicit);
    }
    match bounded_file_source::subject_file(info) {
        Some(path) => bounded_file_source::load(info, path, environment, unusable),
        None if cfg!(feature = "fips") && source.environment_id.is_some() => {
            Err(GoogleLoginFailure::new(unusable, AWS_SOURCE_IN_FIPS))
        }
        None => build(SdkCredentials::ExternalAccount(info.sdk.clone()), unusable),
    }
}
