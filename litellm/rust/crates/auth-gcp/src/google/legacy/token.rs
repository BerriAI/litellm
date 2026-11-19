//! The token exchange the two backwards-compat logins share, [`super::certificate_source`] and
//! [`super::external_account_authorized_user`]: one form POST through litellm's HTTP client, the
//! token it answers, and that token kept until shortly before it expires. It exists only for them:
//! delete it when both are gone.

use std::{
    sync::Arc,
    time::{Duration, Instant},
};

use google_cloud_auth::credentials::{
    CacheableResource, Credentials, CredentialsProvider, EntityTag, impersonated,
};
use google_cloud_auth::errors::CredentialsError;
use http::{Extensions, HeaderMap, HeaderValue};
use litellm_auth_types::{Error, ErrorDetail};
use litellm_secret_value::SecretValue;
use serde::Deserialize;

use crate::google::{
    credentials::target_principal,
    environment::unreachable,
    failure::{GoogleLoginFailure, PythonException},
};
use crate::{GcpAuthFuture, GcpTokenSource, identity::SdkCredentials};

/// How long before its expiry a held token is fetched anew, as Google's Rust SDK refreshes.
const REFRESH_SLACK: Duration = Duration::from_secs(240);

/// A token endpoint's answer to a grant.
#[derive(Deserialize, veil::Redact)]
pub(crate) struct Granted {
    #[redact(with = "[REDACTED]")]
    pub(crate) access_token: String,
    pub(crate) expires_in: Option<u64>,
    #[redact(with = "[REDACTED]")]
    pub(crate) refresh_token: Option<String>,
}

/// Posts a grant and reads its answer: a `TransportError` when the endpoint cannot be reached,
/// google-auth's `OAuthError` when it refuses, a `RefreshError` when it answers no token.
pub(crate) async fn exchange(
    request: reqwest::RequestBuilder,
) -> Result<Granted, GoogleLoginFailure> {
    let response = request.send().await.map_err(unreachable)?;
    let status = response.status();
    let body = response.text().await.map_err(unreachable)?;
    if !status.is_success() {
        return Err(GoogleLoginFailure::new(
            PythonException::OAuth,
            format!("The token endpoint refused the grant with HTTP {status}: {body}"),
        ));
    }
    serde_json::from_str(&body).map_err(|_| {
        GoogleLoginFailure::new(
            PythonException::Refresh,
            "The token endpoint's answer has no access_token.",
        )
    })
}

/// The token a backwards-compat login fetched last and when it stops being usable.
#[derive(Default)]
pub(crate) struct HeldToken(tokio::sync::Mutex<Option<(SecretValue, Option<Instant>)>>);

impl HeldToken {
    /// The held token while it is usable, else the one `fetch` grants.
    pub(crate) async fn get_or_fetch(
        &self,
        fetch: impl Future<Output = Result<Granted, GoogleLoginFailure>>,
    ) -> Result<SecretValue, GoogleLoginFailure> {
        let mut held = self.0.lock().await;
        if let Some((token, until)) = held.as_ref()
            && until.is_none_or(|until| Instant::now() < until)
        {
            return Ok(token.clone());
        }
        let granted = fetch.await?;
        let token = SecretValue::new(granted.access_token);
        let until = granted.expires_in.map(|seconds| {
            Instant::now() + Duration::from_secs(seconds).saturating_sub(REFRESH_SLACK)
        });
        *held = Some((token.clone(), until));
        Ok(token)
    }
}

/// A backwards-compat login: something that grants a Google access token.
pub(crate) trait CompatLogin: Send + Sync + 'static {
    fn token(&self) -> impl Future<Output = Result<SecretValue, GoogleLoginFailure>> + Send;
}

/// A backwards-compat login, as the shared loader's token sources and as the source credentials
/// Google's Rust SDK impersonates a service account with.
pub(crate) struct Compat<T>(pub(crate) Arc<T>);

impl<T> std::fmt::Debug for Compat<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Compat").finish_non_exhaustive()
    }
}

impl<T: CompatLogin> GcpTokenSource for Compat<T> {
    fn project_id(&self) -> GcpAuthFuture<'_, String> {
        Box::pin(async {
            Err(Error::CredentialAcquisition(ErrorDetail::Unresolved {
                subject: "GCP resource project",
            }))
        })
    }

    fn token(&self) -> GcpAuthFuture<'_, SecretValue> {
        Box::pin(async { self.0.token().await.map_err(Error::from) })
    }
}

impl<T: CompatLogin> CredentialsProvider for Compat<T> {
    async fn headers(
        &self,
        _extensions: Extensions,
    ) -> Result<CacheableResource<HeaderMap>, CredentialsError> {
        let token = self.0.token().await.map_err(|failure| {
            let transient = failure.exception == PythonException::Transport;
            CredentialsError::from_source(transient, failure)
        })?;
        let value = HeaderValue::from_str(&format!("Bearer {}", token.expose()))
            .map_err(|error| CredentialsError::from_source(false, error))?;
        Ok(CacheableResource::New {
            entity_tag: EntityTag::new(),
            data: HeaderMap::from_iter([(http::header::AUTHORIZATION, value)]),
        })
    }

    async fn universe_domain(&self) -> Option<String> {
        None
    }
}

/// The SDK credentials that impersonate the service account `url` names with `login`'s token,
/// for `lifetime` when the file sets one.
pub(crate) fn impersonating<T: CompatLogin>(
    login: Arc<T>,
    url: &str,
    lifetime: Option<u64>,
) -> Result<SdkCredentials, GoogleLoginFailure> {
    let principal = target_principal(url).ok_or_else(|| {
        GoogleLoginFailure::new(
            PythonException::Refresh,
            "The service account impersonation URL names no service account.",
        )
    })?;
    let builder = impersonated::Builder::from_source_credentials(Credentials::from(Compat(login)))
        .with_target_principal(principal);
    Ok(SdkCredentials::ImpersonatedFrom(match lifetime {
        Some(seconds) => builder.with_lifetime(Duration::from_secs(seconds)),
        None => builder,
    }))
}
