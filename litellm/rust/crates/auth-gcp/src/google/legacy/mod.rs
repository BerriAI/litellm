//! Backwards compatibility only. Every file here keeps working a login path that Python's
//! google-auth supports and Google's Rust SDK (`google_cloud_auth` 1.16) does not, and says which.
//! Delete each file once the SDK covers its path. The rest of the Google login reaches this module
//! only through the functions below.

mod certificate_source;
mod executable_source;
mod external_account_authorized_user;
mod impersonation;
mod token;

use std::sync::Arc;

pub(crate) use external_account_authorized_user::ExternalAuthorizedUserInfo;

use super::{
    credentials::{CertificateSource, ExecutableSource, ExternalAccountInfo},
    environment::GoogleEnvironment,
    failure::GoogleLoginFailure,
};
use crate::{GcpTokenSource, identity::SdkCredentials};

/// An external account whose `credential_source` is an x509 certificate.
pub(crate) fn certificate_source(
    info: &ExternalAccountInfo,
    certificate: &CertificateSource,
    environment: Arc<dyn GoogleEnvironment>,
) -> Result<Arc<dyn GcpTokenSource>, GoogleLoginFailure> {
    certificate_source::load(info, certificate, environment)
}

/// An external account whose `credential_source` is a credential helper executable, its
/// configuration refused as `explicit` credentials or application default credentials are.
pub(crate) fn executable_source(
    info: &ExternalAccountInfo,
    executable: &ExecutableSource,
    environment: &Arc<dyn GoogleEnvironment>,
    explicit: bool,
) -> Result<Arc<dyn GcpTokenSource>, GoogleLoginFailure> {
    executable_source::load(info, executable, environment, explicit)
}

/// A workforce identity user's `external_account_authorized_user` credentials.
pub(crate) fn external_account_authorized_user(
    info: &ExternalAuthorizedUserInfo,
    environment: Arc<dyn GoogleEnvironment>,
) -> Result<Arc<dyn GcpTokenSource>, GoogleLoginFailure> {
    Ok(Arc::new(token::Compat(
        external_account_authorized_user::load(info, environment)?,
    )))
}

/// The SDK credentials that impersonate the service account `url` names with a workforce identity
/// user's token.
pub(crate) fn impersonating_external_account_authorized_user(
    info: &ExternalAuthorizedUserInfo,
    url: &str,
    environment: Arc<dyn GoogleEnvironment>,
) -> Result<SdkCredentials, GoogleLoginFailure> {
    token::impersonating(
        external_account_authorized_user::load(info, environment)?,
        url,
        None,
    )
}

/// Whether Google's Rust SDK impersonates at `url` itself, which it does only on its default
/// host.
pub(crate) fn sdk_impersonates_at(url: &str) -> bool {
    impersonation::sdk_calls(url)
}

/// `source` impersonating the service account at `url`, an endpoint Google's Rust SDK would not
/// call, for `lifetime_seconds` when the account sets it.
pub(crate) fn impersonated_at(
    source: Arc<dyn GcpTokenSource>,
    url: &str,
    lifetime_seconds: Option<u64>,
    environment: Arc<dyn GoogleEnvironment>,
) -> Arc<dyn GcpTokenSource> {
    Arc::new(token::Compat(Arc::new(
        impersonation::CustomImpersonation::new(source, url, lifetime_seconds, environment),
    )))
}
