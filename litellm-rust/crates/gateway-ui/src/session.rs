use std::sync::Arc;

use axum::{Json, extract::State as ExtractState};
use base64::{Engine, engine::general_purpose::URL_SAFE_NO_PAD};
use governor::DefaultDirectRateLimiter;
use jsonwebtoken::{EncodingKey, Header, encode};
use litellm_gateway_auth::{UI_CSRF_KEY, UiAuthError, UiAuthSession, UiCredentials, UiSession};
use serde::Serialize;
use time::{Duration, OffsetDateTime};
use tower_cookies::{Cookie, Cookies};
use tower_sessions::{Expiry, Session, cookie::SameSite};

use crate::Error;

pub struct State {
    pub signing_key: EncodingKey,
    pub login_limit: DefaultDirectRateLimiter,
    pub secure_cookies: bool,
}

#[derive(Serialize)]
struct Claims<'a> {
    key: &'a str,
    user_id: &'a str,
    user_role: &'static str,
    login_method: &'static str,
    premium_user: bool,
    auth_header_name: &'static str,
    server_root_path: &'static str,
    exp: i64,
}

#[derive(Serialize)]
pub struct LoginResponse {
    redirect_url: &'static str,
    token: String,
}

#[derive(Serialize)]
pub struct SessionInfo {
    user_id: String,
    user_role: &'static str,
}

#[derive(Serialize)]
pub struct LogoutResponse {
    message: &'static str,
}

#[derive(Serialize)]
pub struct Discovery {
    server_root_path: &'static str,
    proxy_base_url: Option<&'static str>,
    auto_redirect_to_sso: bool,
    admin_ui_disabled: bool,
    sso_configured: bool,
    hide_default_credentials_hint: bool,
    is_control_plane: bool,
}

pub async fn login(
    ExtractState(state): ExtractState<Arc<State>>,
    mut auth: UiAuthSession,
    session: Session,
    cookies: Cookies,
    Json(credentials): Json<UiCredentials>,
) -> Result<Json<LoginResponse>, Error> {
    state.login_limit.check().map_err(|_| Error::RateLimited)?;
    let user = auth
        .authenticate(credentials)
        .await?
        .ok_or(Error::InvalidCredentials)?;
    let expires = OffsetDateTime::now_utc() + Duration::hours(24);
    let csrf = URL_SAFE_NO_PAD.encode(rand::random::<[u8; 32]>());
    let token = encode(
        &Header::default(),
        &Claims {
            key: &csrf,
            user_id: &user.username,
            user_role: "proxy_admin",
            login_method: "username_password",
            premium_user: false,
            auth_header_name: "Authorization",
            server_root_path: "",
            exp: expires.unix_timestamp(),
        },
        &state.signing_key,
    )?;
    auth.logout().await?;
    auth.login(&user).await?;
    session.cycle_id().await?;
    session.insert(UI_CSRF_KEY, csrf).await?;
    session.set_expiry(Some(Expiry::AtDateTime(expires)));
    cookies.add(
        Cookie::build(("token", token.clone()))
            .path("/ui")
            .same_site(SameSite::Strict)
            .secure(state.secure_cookies)
            .expires(expires)
            .build(),
    );
    Ok(Json(LoginResponse {
        redirect_url: "/ui/?login=success",
        token,
    }))
}

pub async fn info(session: Result<UiSession, UiAuthError>) -> Result<Json<SessionInfo>, Error> {
    let session = session?;
    Ok(Json(SessionInfo {
        user_id: session.user.username,
        user_role: "proxy_admin",
    }))
}

pub async fn logout(
    validated: Result<UiSession, UiAuthError>,
    mut auth: UiAuthSession,
    cookies: Cookies,
) -> Result<Json<LogoutResponse>, Error> {
    let _ = validated?;
    auth.logout().await?;
    cookies.add(
        Cookie::build(("token", ""))
            .path("/ui")
            .max_age(Duration::ZERO)
            .build(),
    );
    Ok(Json(LogoutResponse {
        message: "Session revoked.",
    }))
}

pub async fn discovery() -> Json<Discovery> {
    Json(Discovery {
        server_root_path: "",
        proxy_base_url: None,
        auto_redirect_to_sso: false,
        admin_ui_disabled: false,
        sso_configured: false,
        hide_default_credentials_hint: true,
        is_control_plane: false,
    })
}
