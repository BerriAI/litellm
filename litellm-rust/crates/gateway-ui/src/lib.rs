mod assets;
mod error;
mod session;

use std::{num::NonZeroU32, path::Path, sync::Arc};

use axum::{
    Router,
    http::{HeaderValue, header},
    routing::{get, post},
};
use axum_login::AuthManagerLayerBuilder;
use governor::{Quota, RateLimiter};
use jsonwebtoken::EncodingKey;
use litellm_gateway_auth::UiBackend;
use tower_http::set_header::SetResponseHeaderLayer;
use tower_sessions::{SessionManagerLayer, SessionStore, cookie::SameSite};

pub use error::Error;

pub fn router(
    directory: impl AsRef<Path>,
    backend: UiBackend,
    store: impl SessionStore + Clone,
    secure_cookies: bool,
) -> Router {
    let sessions = SessionManagerLayer::new(store)
        .with_name("litellm_session")
        .with_http_only(true)
        .with_secure(secure_cookies)
        .with_same_site(SameSite::Strict);
    let auth = AuthManagerLayerBuilder::new(backend, sessions).build();
    let state = Arc::new(session::State {
        signing_key: EncodingKey::from_secret(&rand::random::<[u8; 32]>()),
        login_limit: RateLimiter::direct(Quota::per_minute(NonZeroU32::new(10).unwrap())),
        secure_cookies,
    });
    Router::new()
        .route("/v2/login", post(session::login))
        .route("/session/info", get(session::info))
        .route("/session/logout", post(session::logout))
        .layer(auth)
        .route("/.well-known/litellm-ui-config", get(session::discovery))
        .route(
            "/litellm/.well-known/litellm-ui-config",
            get(session::discovery),
        )
        .layer(axum::extract::DefaultBodyLimit::max(16 * 1024))
        .layer(SetResponseHeaderLayer::overriding(
            header::CACHE_CONTROL,
            HeaderValue::from_static("no-store"),
        ))
        .with_state(state)
        .merge(assets::router(directory.as_ref()))
}
