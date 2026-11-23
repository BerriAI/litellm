//! Backwards compatibility only, with google-auth's `external_account_authorized_user`
//! credentials: a workforce identity user's refresh token, exchanged once per token at the STS
//! endpoint its file names. Google's Rust SDK (`google_cloud_auth` 1.16) has no such credential
//! type. Delete this module, and [`super::token`] once the certificate source is gone too, when
//! the SDK adds one.

use std::sync::{Arc, Mutex, PoisonError};

use base64::{Engine, engine::general_purpose::STANDARD};
use litellm_secret_value::SecretValue;
use serde::Deserialize;
use time::{OffsetDateTime, PrimitiveDateTime, macros::format_description};

use super::token::{CompatLogin, HeldToken, exchange};
use crate::google::{
    environment::GoogleEnvironment,
    failure::{GoogleLoginFailure, PythonException},
};

const CANNOT_REFRESH: &str = "The external account authorized user credentials need refresh_token, token_url, client_id and client_secret to refresh.";
/// `google.auth._helpers.REFRESH_THRESHOLD`: a token this close to its expiry is not valid.
const REFRESH_THRESHOLD: time::Duration = time::Duration::seconds(225);

/// A workforce identity user's file, as `Credentials.from_info` reads it.
#[derive(Clone, Default, Deserialize, PartialEq, veil::Redact)]
pub(crate) struct ExternalAuthorizedUserInfo {
    #[redact(with = "[REDACTED]")]
    pub(crate) token: Option<String>,
    pub(crate) expiry: Option<String>,
    #[redact(with = "[REDACTED]")]
    pub(crate) refresh_token: Option<String>,
    pub(crate) token_url: Option<String>,
    pub(crate) client_id: Option<String>,
    #[redact(with = "[REDACTED]")]
    pub(crate) client_secret: Option<String>,
}

/// What a refresh needs: the refresh token, the STS endpoint and the client it authenticates as.
struct Refresh {
    refresh_token: String,
    token_url: String,
    client_id: String,
    client_secret: String,
}

/// A workforce identity user's login: its refresh token, rotated when STS answers a new one.
pub(crate) struct ExternalAuthorizedUser {
    refresh: Option<Refresh>,
    rotated: Mutex<Option<String>>,
    held: HeldToken,
    environment: Arc<dyn GoogleEnvironment>,
}

/// `datetime.strptime(expiry.rstrip("Z").split(".")[0], "%Y-%m-%dT%H:%M:%S")`, a naive UTC time.
fn expiry(text: &str) -> Option<OffsetDateTime> {
    let seconds = text.trim_end_matches('Z').split('.').next()?;
    PrimitiveDateTime::parse(
        seconds,
        format_description!("[year]-[month]-[day]T[hour]:[minute]:[second]"),
    )
    .ok()
    .map(PrimitiveDateTime::assume_utc)
}

/// `Credentials.from_info`. The file's own token only decides whether the credentials load,
/// since the Vertex login refreshes before it takes a token: they load when they can refresh or
/// hold a token that is still valid.
pub(crate) fn load(
    info: &ExternalAuthorizedUserInfo,
    environment: Arc<dyn GoogleEnvironment>,
) -> Result<Arc<ExternalAuthorizedUser>, GoogleLoginFailure> {
    let unloadable = || {
        GoogleLoginFailure::new(
            PythonException::DefaultCredentials,
            "The external account authorized user credentials cannot be used.",
        )
    };
    let expiry = match info.expiry.as_deref().filter(|text| !text.is_empty()) {
        Some(text) => Some(expiry(text).ok_or_else(unloadable)?),
        None => None,
    };
    let given = |field: &Option<String>| field.clone().filter(|text| !text.is_empty());
    let refresh = match (
        given(&info.refresh_token),
        given(&info.token_url),
        given(&info.client_id),
        given(&info.client_secret),
    ) {
        (Some(refresh_token), Some(token_url), Some(client_id), Some(client_secret)) => {
            Some(Refresh {
                refresh_token,
                token_url,
                client_id,
                client_secret,
            })
        }
        _ => None,
    };
    let valid_token = info.token.is_some()
        && expiry.is_none_or(|expiry| OffsetDateTime::now_utc() < expiry - REFRESH_THRESHOLD);
    if refresh.is_none() && !valid_token {
        return Err(unloadable());
    }
    Ok(Arc::new(ExternalAuthorizedUser {
        refresh,
        rotated: Mutex::new(None),
        held: HeldToken::default(),
        environment,
    }))
}

impl ExternalAuthorizedUser {
    fn rotated(&self) -> std::sync::MutexGuard<'_, Option<String>> {
        self.rotated.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// `sts.Client.refresh_token`: the refresh grant, with HTTP basic client authentication.
    async fn refresh(&self, refresh: &Refresh) -> Result<SecretValue, GoogleLoginFailure> {
        self.held
            .get_or_fetch(async {
                let refresh_token = self
                    .rotated()
                    .clone()
                    .unwrap_or_else(|| refresh.refresh_token.clone());
                let basic =
                    STANDARD.encode(format!("{}:{}", refresh.client_id, refresh.client_secret));
                let granted = exchange(
                    self.environment
                        .http_client(&refresh.token_url, None)?
                        .post(&refresh.token_url)
                        .header(http::header::AUTHORIZATION, format!("Basic {basic}"))
                        .form(&[
                            ("grant_type", "refresh_token"),
                            ("refresh_token", refresh_token.as_str()),
                        ]),
                )
                .await?;
                if let Some(rotated) = &granted.refresh_token {
                    *self.rotated() = Some(rotated.clone());
                }
                Ok(granted)
            })
            .await
    }
}

impl CompatLogin for ExternalAuthorizedUser {
    async fn token(&self) -> Result<SecretValue, GoogleLoginFailure> {
        match &self.refresh {
            Some(refresh) => self.refresh(refresh).await,
            None => Err(GoogleLoginFailure::new(
                PythonException::Refresh,
                CANNOT_REFRESH,
            )),
        }
    }
}
