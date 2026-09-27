use base64::{Engine, engine::general_purpose::URL_SAFE_NO_PAD};
use serde_json::{Value, json};
use tempfile::TempDir;
use time::{Duration, OffsetDateTime};

use axum::{
    Router,
    body::{Body, to_bytes},
    http::{Request, StatusCode, header},
    response::Response,
};
use rstest::{fixture, rstest};
use tower::ServiceExt;
use tower_cookies::Cookie;
use tower_sessions::{
    MemoryStore, SessionStore,
    session::{Id, Record},
};

use litellm_auth_types::SecretValue;
use litellm_gateway_auth::UiBackend;

struct App {
    router: Router,
    store: MemoryStore,
    directory: TempDir,
}

#[fixture]
fn app() -> App {
    let directory = TempDir::new().unwrap();
    let export = directory.path().join("public");
    std::fs::create_dir_all(export.join("login")).unwrap();
    std::fs::create_dir_all(export.join("_next/static")).unwrap();
    std::fs::create_dir_all(export.join("assets/logos")).unwrap();
    std::fs::write(export.join("assets/logos/litellm_logo.jpg"), "logo bytes").unwrap();
    std::fs::write(export.join("favicon.ico"), "icon bytes").unwrap();
    std::fs::write(directory.path().join("outside.txt"), "private file").unwrap();
    std::fs::write(export.join("index.html"), "dashboard").unwrap();
    std::fs::write(export.join("login/index.html"), "login page").unwrap();
    std::fs::write(export.join("_next/static/app.js"), "window.app = true;").unwrap();
    let store = MemoryStore::default();
    let backend = UiBackend::new("admin".into(), SecretValue::new("test-password")).unwrap();
    let router = litellm_gateway_ui::router(export, backend, store.clone(), true);
    App {
        router,
        store,
        directory,
    }
}

async fn body(response: Response) -> Value {
    serde_json::from_slice(&to_bytes(response.into_body(), 65536).await.unwrap()).unwrap()
}

async fn login(app: &App, cookie: Option<&str>) -> Response {
    app.router
        .clone()
        .oneshot(
            Request::post("/v2/login")
                .header(header::CONTENT_TYPE, "application/json")
                .header(header::COOKIE, cookie.unwrap_or(""))
                .body(Body::from(
                    json!({"username": "admin", "password": "test-password"}).to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap()
}

fn session_cookie(response: &Response) -> Cookie<'static> {
    response
        .headers()
        .get_all(header::SET_COOKIE)
        .iter()
        .map(|value| Cookie::parse(value.to_str().unwrap().to_owned()).unwrap())
        .find(|cookie| cookie.name() == "litellm_session")
        .unwrap()
}

async fn claims(response: Response) -> Value {
    let json = body(response).await;
    let token = json["token"].as_str().unwrap();
    serde_json::from_slice(
        &URL_SAFE_NO_PAD
            .decode(token.split('.').nth(1).unwrap())
            .unwrap(),
    )
    .unwrap()
}

async fn protected(app: &App, path: &str, method: &str, cookie: &str, csrf: &str) -> Response {
    app.router
        .clone()
        .oneshot(
            Request::builder()
                .uri(path)
                .method(method)
                .header(header::COOKIE, cookie)
                .header(header::AUTHORIZATION, format!("Bearer {csrf}"))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap()
}

#[rstest]
#[case::dashboard("/ui/", "dashboard", "text/html")]
#[case::login("/ui/login/", "login page", "text/html")]
#[case::asset_alias(
    "/litellm-asset-prefix/_next/static/app.js",
    "window.app = true;",
    "text/javascript"
)]
#[case::root_assets("/_next/static/app.js", "window.app = true;", "text/javascript")]
#[case::nested_assets("/ui/_next/static/app.js", "window.app = true;", "text/javascript")]
#[case::logo("/get_image", "logo bytes", "image/jpeg")]
#[case::favicon("/get_favicon", "icon bytes", "image/x-icon")]
#[tokio::test]
async fn serves_export_without_auth(
    app: App,
    #[case] path: &str,
    #[case] expected: &str,
    #[case] mime: &str,
) {
    let response = app
        .router
        .oneshot(Request::get(path).body(Body::empty()).unwrap())
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    assert!(
        response.headers()[header::CONTENT_TYPE]
            .to_str()
            .unwrap()
            .starts_with(mime)
    );
    assert_eq!(
        to_bytes(response.into_body(), 65536).await.unwrap(),
        expected
    );
}

#[rstest]
#[case::root("/ui?login=success", "/ui/?login=success")]
#[case::nested("/ui/login?redirect_to=%2Fui", "/ui/login/?redirect_to=%2Fui")]
#[tokio::test]
async fn directory_redirects_preserve_prefix_and_query(
    app: App,
    #[case] path: &str,
    #[case] location: &str,
) {
    let response = app
        .router
        .oneshot(Request::get(path).body(Body::empty()).unwrap())
        .await
        .unwrap();
    assert!(response.status().is_redirection());
    assert_eq!(response.headers()[header::LOCATION], location);
}

#[rstest]
#[case::missing("/ui/no-such-page")]
#[case::outside_ui("/v1/models")]
#[case::traversal("/ui/%2e%2e/outside.txt")]
#[case::encoded_separator("/ui/..%2foutside.txt")]
#[tokio::test]
async fn missing_paths_never_fall_back_to_dashboard(app: App, #[case] path: &str) {
    let response = app
        .router
        .oneshot(Request::get(path).body(Body::empty()).unwrap())
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::NOT_FOUND);
    assert!(
        !to_bytes(response.into_body(), 65536)
            .await
            .unwrap()
            .windows(9)
            .any(|part| part == b"dashboard")
    );
}

#[rstest]
#[case::root("/.well-known/litellm-ui-config")]
#[case::compatibility("/litellm/.well-known/litellm-ui-config")]
#[tokio::test]
async fn discovery_describes_local_login(app: App, #[case] path: &str) {
    let response = app
        .router
        .oneshot(Request::get(path).body(Body::empty()).unwrap())
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(response.headers()[header::CACHE_CONTROL], "no-store");
    let config = body(response).await;
    assert_eq!(config["sso_configured"], false);
    assert_eq!(config["hide_default_credentials_hint"], true);
    assert_eq!(config["server_root_path"], "");
}

#[rstest]
#[tokio::test]
async fn logo_discovery_points_to_served_image(app: App) {
    let response = app
        .router
        .clone()
        .oneshot(Request::get("/get_logo_url").body(Body::empty()).unwrap())
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let config = body(response).await;
    let response = app
        .router
        .oneshot(
            Request::get(config["logo_url"].as_str().unwrap())
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        to_bytes(response.into_body(), 65536).await.unwrap(),
        "logo bytes"
    );
}

#[rstest]
#[tokio::test]
async fn login_sets_bounded_session_and_requires_cookie_with_csrf(app: App) {
    let started = OffsetDateTime::now_utc();
    let response = login(&app, None).await;
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(response.headers()[header::CACHE_CONTROL], "no-store");
    let cookie = session_cookie(&response);
    assert_eq!(cookie.http_only(), Some(true));
    assert_eq!(cookie.secure(), Some(true));
    assert_eq!(
        cookie.same_site(),
        Some(tower_cookies::cookie::SameSite::Strict)
    );
    assert_eq!(cookie.path(), Some("/"));
    let display_cookie = response
        .headers()
        .get_all(header::SET_COOKIE)
        .iter()
        .map(|value| Cookie::parse(value.to_str().unwrap().to_owned()).unwrap())
        .find(|cookie| cookie.name() == "token")
        .unwrap();
    assert_eq!(display_cookie.path(), Some("/ui"));
    assert_eq!(display_cookie.secure(), Some(true));
    assert_ne!(display_cookie.http_only(), Some(true));
    assert_eq!(
        display_cookie.same_site(),
        Some(tower_cookies::cookie::SameSite::Strict)
    );
    let json = body(response).await;
    assert_eq!(json["redirect_url"], "/ui/?login=success");
    assert_eq!(json["token"], display_cookie.value());
    let token: Value = serde_json::from_slice(
        &URL_SAFE_NO_PAD
            .decode(display_cookie.value().split('.').nth(1).unwrap())
            .unwrap(),
    )
    .unwrap();
    let csrf = token["key"].as_str().unwrap();
    assert_eq!(token["user_id"], "admin");
    assert_eq!(token["user_role"], "proxy_admin");
    assert_eq!(token["auth_header_name"], "Authorization");
    assert_ne!(csrf, cookie.value());
    assert_ne!(csrf, "test-password");
    let record = app
        .store
        .load(&cookie.value().parse::<Id>().unwrap())
        .await
        .unwrap()
        .unwrap();
    assert_eq!(token["exp"], record.expiry_date.unix_timestamp());
    assert_eq!(
        display_cookie.expires_datetime().unwrap().unix_timestamp(),
        record.expiry_date.unix_timestamp()
    );
    assert!(record.expiry_date >= started + Duration::hours(24));
    assert!(record.expiry_date <= OffsetDateTime::now_utc() + Duration::hours(24));
    let response = protected(&app, "/session/info", "GET", &cookie.to_string(), csrf).await;
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        body(response).await,
        json!({"user_id": "admin", "user_role": "proxy_admin"})
    );
    assert_eq!(
        app.store
            .load(&record.id)
            .await
            .unwrap()
            .unwrap()
            .expiry_date,
        record.expiry_date
    );
    assert_eq!(
        protected(&app, "/session/info", "GET", "", csrf)
            .await
            .status(),
        StatusCode::UNAUTHORIZED
    );
    assert_eq!(
        protected(&app, "/session/info", "GET", &cookie.to_string(), "wrong")
            .await
            .status(),
        StatusCode::UNAUTHORIZED
    );
}

#[rstest]
#[tokio::test]
async fn logout_revokes_only_the_presented_session(app: App) {
    let first = login(&app, None).await;
    let first_cookie = session_cookie(&first);
    let first_claims = claims(first).await;
    let first_csrf = first_claims["key"].as_str().unwrap();
    let second = login(&app, None).await;
    let second_cookie = session_cookie(&second);
    let second_claims = claims(second).await;
    let second_csrf = second_claims["key"].as_str().unwrap();
    let rejected = protected(
        &app,
        "/session/logout",
        "POST",
        &first_cookie.to_string(),
        second_csrf,
    )
    .await;
    assert_eq!(rejected.status(), StatusCode::UNAUTHORIZED);
    let response = protected(
        &app,
        "/session/logout",
        "POST",
        &first_cookie.to_string(),
        first_csrf,
    )
    .await;
    assert_eq!(response.status(), StatusCode::OK);
    let removed = response
        .headers()
        .get_all(header::SET_COOKIE)
        .iter()
        .map(|value| Cookie::parse(value.to_str().unwrap().to_owned()).unwrap())
        .find(|cookie| cookie.name() == "token")
        .unwrap();
    assert_eq!(removed.value(), "");
    assert_eq!(removed.path(), Some("/ui"));
    assert_eq!(removed.max_age(), Some(Duration::ZERO));
    assert_eq!(session_cookie(&response).max_age(), Some(Duration::ZERO));
    assert_eq!(
        protected(
            &app,
            "/session/info",
            "GET",
            &first_cookie.to_string(),
            first_csrf
        )
        .await
        .status(),
        StatusCode::UNAUTHORIZED
    );
    assert_eq!(
        protected(
            &app,
            "/session/info",
            "GET",
            &second_cookie.to_string(),
            second_csrf
        )
        .await
        .status(),
        StatusCode::OK
    );
}

#[rstest]
#[tokio::test]
async fn expired_sessions_are_rejected(app: App) {
    let response = login(&app, None).await;
    let cookie = session_cookie(&response);
    let token = claims(response).await;
    let record = app
        .store
        .load(&cookie.value().parse().unwrap())
        .await
        .unwrap()
        .unwrap();
    app.store
        .save(&Record {
            expiry_date: OffsetDateTime::now_utc() - Duration::seconds(1),
            ..record
        })
        .await
        .unwrap();
    assert_eq!(
        protected(
            &app,
            "/session/info",
            "GET",
            &cookie.to_string(),
            token["key"].as_str().unwrap()
        )
        .await
        .status(),
        StatusCode::UNAUTHORIZED
    );
}

#[rstest]
#[tokio::test]
async fn reauthentication_rotates_and_revokes_the_previous_session(app: App) {
    let first = login(&app, None).await;
    let first_cookie = session_cookie(&first);
    let first_claims = claims(first).await;
    let second = login(&app, Some(&first_cookie.to_string())).await;
    assert_eq!(second.status(), StatusCode::OK);
    let second_cookie = session_cookie(&second);
    assert_ne!(first_cookie.value(), second_cookie.value());
    let second_claims = claims(second).await;
    assert_ne!(first_claims["key"], second_claims["key"]);
    assert_eq!(
        protected(
            &app,
            "/session/info",
            "GET",
            &second_cookie.to_string(),
            second_claims["key"].as_str().unwrap()
        )
        .await
        .status(),
        StatusCode::OK
    );
    assert_eq!(
        protected(
            &app,
            "/session/info",
            "GET",
            &first_cookie.to_string(),
            first_claims["key"].as_str().unwrap()
        )
        .await
        .status(),
        StatusCode::UNAUTHORIZED
    );
}

#[rstest]
#[tokio::test]
async fn failed_logins_are_generic_and_rate_limited(app: App) {
    for attempt in 0..11 {
        let response = app.router.clone().oneshot(Request::post("/v2/login")
            .header(header::CONTENT_TYPE, "application/json")
            .body(Body::from(json!({"username": format!("unknown-{attempt}"), "password": "wrong-password"}).to_string())).unwrap()).await.unwrap();
        if attempt < 10 {
            assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
            assert_eq!(
                body(response).await["error"]["message"],
                "Invalid username or password"
            );
        } else {
            assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
            assert!(response.headers().contains_key(header::RETRY_AFTER));
        }
    }
}

#[rstest]
#[tokio::test]
async fn login_rejects_browser_form_posts(app: App) {
    let response = app
        .router
        .oneshot(
            Request::post("/v2/login")
                .header(header::CONTENT_TYPE, "application/x-www-form-urlencoded")
                .body(Body::from("username=admin&password=test-password"))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::UNSUPPORTED_MEDIA_TYPE);
    assert!(!response.headers().contains_key(header::SET_COOKIE));
}

#[rstest]
#[case::missing(None)]
#[case::empty(Some(""))]
#[case::wrong_scheme(Some("Basic invalid"))]
#[case::empty_bearer(Some("Bearer "))]
#[case::wrong_bearer(Some("Bearer invalid"))]
#[tokio::test]
async fn session_requires_csrf_header(app: App, #[case] authorization: Option<&str>) {
    let response = login(&app, None).await;
    let cookie = session_cookie(&response);
    let request = Request::get("/session/info").header(header::COOKIE, cookie.to_string());
    let request = match authorization {
        Some(value) => request.header(header::AUTHORIZATION, value),
        None => request,
    };
    let response = app
        .router
        .oneshot(request.body(Body::empty()).unwrap())
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    assert_eq!(response.headers()[header::CACHE_CONTROL], "no-store");
}

#[rstest]
#[case::missing("")]
#[case::malformed("litellm_session=invalid")]
#[case::unknown("litellm_session=AAAAAAAAAAAAAAAAAAAAAA")]
#[tokio::test]
async fn logout_requires_authenticated_session(app: App, #[case] cookie: &str) {
    let response = protected(&app, "/session/logout", "POST", cookie, "invalid").await;
    assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
}

#[rstest]
#[case::wrong_password("admin", "wrong")]
#[case::unknown_user("unknown", "test-password")]
#[tokio::test]
async fn failed_login_preserves_existing_session(
    app: App,
    #[case] username: &str,
    #[case] password: &str,
) {
    let original = login(&app, None).await;
    let cookie = session_cookie(&original);
    let token = claims(original).await;
    let response = app
        .router
        .clone()
        .oneshot(
            Request::post("/v2/login")
                .header(header::CONTENT_TYPE, "application/json")
                .header(header::COOKIE, cookie.to_string())
                .body(Body::from(
                    json!({"username": username, "password": password}).to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    assert_eq!(
        body(response).await["error"]["message"],
        "Invalid username or password"
    );
    assert_eq!(
        protected(
            &app,
            "/session/info",
            "GET",
            &cookie.to_string(),
            token["key"].as_str().unwrap()
        )
        .await
        .status(),
        StatusCode::OK
    );
}

#[rstest]
#[case::malformed("{".into(), StatusCode::BAD_REQUEST)]
#[case::missing_password(r#"{"username":"admin"}"#.into(), StatusCode::UNPROCESSABLE_ENTITY)]
#[case::oversized(json!({"username": "admin", "password": "x".repeat(16 * 1024)}).to_string(), StatusCode::PAYLOAD_TOO_LARGE)]
#[tokio::test]
async fn invalid_login_payloads_do_not_create_sessions(
    app: App,
    #[case] payload: String,
    #[case] status: StatusCode,
) {
    let response = app
        .router
        .oneshot(
            Request::post("/v2/login")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(payload))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), status);
    assert!(!response.headers().contains_key(header::SET_COOKIE));
}

#[rstest]
#[tokio::test]
async fn changed_password_revokes_sessions_in_shared_store(app: App) {
    let original = login(&app, None).await;
    let cookie = session_cookie(&original);
    let token = claims(original).await;
    let replaced = litellm_gateway_ui::router(
        app.directory.path().join("public"),
        UiBackend::new("admin".into(), SecretValue::new("new-password")).unwrap(),
        app.store,
        true,
    );
    let response = replaced
        .oneshot(
            Request::get("/session/info")
                .header(header::COOKIE, cookie.to_string())
                .header(
                    header::AUTHORIZATION,
                    format!("Bearer {}", token["key"].as_str().unwrap()),
                )
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
}
