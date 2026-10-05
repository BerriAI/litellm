use std::{convert::Infallible, sync::Arc};

use axum::extract::FromRequestParts;
use axum::http::{header::AUTHORIZATION, request::Parts};
use axum_login::{AuthUser, AuthnBackend, UserId};
use litellm_auth_types::SecretValue;
use serde::Deserialize;
use sha2::{Digest, Sha256};
use subtle::ConstantTimeEq;
use tower_sessions::Session;
use veil::Redact;

use crate::{
    AccessRequest, AuthenticatedRequest, Authentication, AuthenticationMethod, LocalAdministrator,
    NoAdditionalPolicy, Permissions, Principal, PrincipalKind, SystemClock, UiAction, UiAuthError,
    VerifiedIdentity,
};

pub const UI_CSRF_KEY: &str = "litellm.ui.csrf";
pub type UiAuthSession = axum_login::AuthSession<UiBackend>;

#[derive(Clone, Redact)]
pub struct UiUser {
    pub username: String,
    #[redact]
    password_hash: [u8; 32],
}

impl AuthUser for UiUser {
    type Id = String;

    fn id(&self) -> Self::Id {
        self.username.clone()
    }

    fn session_auth_hash(&self) -> &[u8] {
        &self.password_hash
    }
}

#[derive(Deserialize)]
pub struct UiCredentials {
    pub username: String,
    pub password: SecretValue,
}

#[derive(Clone)]
pub struct UiBackend {
    user: UiUser,
}

impl UiBackend {
    pub fn new(username: String, password: SecretValue) -> Result<Self, UiAuthError> {
        if username.trim().is_empty() || password.expose().trim().is_empty() {
            return Err(UiAuthError::Unconfigured);
        }
        Ok(Self {
            user: UiUser {
                username,
                password_hash: Sha256::digest(password.expose()).into(),
            },
        })
    }
}

impl AuthnBackend for UiBackend {
    type User = UiUser;
    type Credentials = UiCredentials;
    type Error = Infallible;

    async fn authenticate(&self, credentials: UiCredentials) -> Result<Option<UiUser>, Infallible> {
        let password_matches = self
            .user
            .password_hash
            .ct_eq(&Sha256::digest(credentials.password.expose()));
        let username_matches =
            Sha256::digest(&self.user.username).ct_eq(&Sha256::digest(credentials.username));
        Ok(bool::from(password_matches & username_matches).then(|| self.user.clone()))
    }

    async fn get_user(&self, user_id: &UserId<Self>) -> Result<Option<UiUser>, Infallible> {
        Ok((user_id == &self.user.username).then(|| self.user.clone()))
    }
}

pub struct UiSession {
    pub user: UiUser,
    pub identity: AuthenticatedRequest,
}

impl<S: Send + Sync> FromRequestParts<S> for UiSession {
    type Rejection = UiAuthError;

    async fn from_request_parts(parts: &mut Parts, state: &S) -> Result<Self, Self::Rejection> {
        let auth = UiAuthSession::from_request_parts(parts, state)
            .await
            .map_err(|_| UiAuthError::Unavailable)?;
        let user = auth.user.ok_or(UiAuthError::Unauthorized)?;
        let session = Session::from_request_parts(parts, state)
            .await
            .map_err(|_| UiAuthError::Unavailable)?;
        let expected = session
            .get::<String>(UI_CSRF_KEY)
            .await?
            .ok_or(UiAuthError::Unauthorized)?;
        let provided = parts
            .headers
            .get(AUTHORIZATION)
            .and_then(|value| value.to_str().ok())
            .and_then(|value| value.strip_prefix("Bearer "))
            .ok_or(UiAuthError::Unauthorized)?;
        if !bool::from(Sha256::digest(expected).ct_eq(&Sha256::digest(provided))) {
            return Err(UiAuthError::Unauthorized);
        }
        let identity = session_identity(&user).await?;
        parts.extensions.insert(identity.clone());
        Ok(Self { user, identity })
    }
}

async fn session_identity(user: &UiUser) -> Result<AuthenticatedRequest, UiAuthError> {
    crate::authentication::resolve(
        VerifiedIdentity {
            principal: Principal::new(
                "litellm:local-ui".into(),
                user.username.clone(),
                PrincipalKind::Human,
            ),
            authentication: Authentication {
                method: AuthenticationMethod::Session,
                verifier: "litellm:local-ui".into(),
                credential_id: user.username.clone(),
                expires_at: None,
            },
            restrictions: Permissions::Only(Arc::from([
                AccessRequest::Ui(UiAction::SessionInfo),
                AccessRequest::Ui(UiAction::Logout),
            ])),
        },
        &LocalAdministrator,
        Arc::new(NoAdditionalPolicy),
        Arc::new(SystemClock),
    )
    .await
    .map_err(|_| UiAuthError::Unauthorized)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::rstest;

    #[rstest]
    #[tokio::test]
    async fn ui_identity_is_scoped_to_session_operations() {
        let backend = UiBackend::new("admin".into(), SecretValue::new("password")).unwrap();
        let user = backend.get_user(&"admin".into()).await.unwrap().unwrap();
        let identity = session_identity(&user).await.unwrap();
        assert_eq!(identity.caller().principal().subject(), user.username);
        assert_eq!(
            identity.caller().authentication().method,
            AuthenticationMethod::Session
        );
        assert!(
            identity
                .authorize(AccessRequest::Ui(UiAction::SessionInfo))
                .await
                .is_ok()
        );
        assert!(
            identity
                .authorize(AccessRequest::Ui(UiAction::Logout))
                .await
                .is_ok()
        );
        assert!(matches!(
            identity
                .authorize(AccessRequest::Model {
                    name: "model".into(),
                    deployment: "provider/model".into(),
                })
                .await,
            Err(crate::Error::Forbidden)
        ));
    }
}
