use axum::{
    Router,
    body::{Body, to_bytes},
    http::{Request, StatusCode, header},
};
use rstest::{fixture, rstest};
use serde_json::Value;
use tempfile::TempDir;
use tower::ServiceExt;

struct App {
    router: Router,
    _directory: TempDir,
}

#[fixture]
fn directory() -> TempDir {
    let directory = TempDir::new().unwrap();
    let export = directory.path().join("public");
    std::fs::create_dir_all(export.join("login")).unwrap();
    std::fs::create_dir_all(export.join("assets")).unwrap();
    std::fs::write(directory.path().join("outside.txt"), "private file").unwrap();
    std::fs::write(export.join("index.html"), "dashboard").unwrap();
    std::fs::write(export.join("login/index.html"), "login page").unwrap();
    std::fs::write(export.join("assets/app.js"), "window.app = true;").unwrap();
    std::fs::write(export.join("styles.css"), "body { color: black; }").unwrap();
    directory
}

#[fixture]
fn app(directory: TempDir) -> App {
    App {
        router: litellm_gateway_ui::static_assets(directory.path().join("public")),
        _directory: directory,
    }
}

#[fixture]
fn dashboard(directory: TempDir) -> App {
    let export = directory.path().join("public");
    std::fs::create_dir_all(export.join("_next/static")).unwrap();
    std::fs::create_dir_all(export.join("assets/logos")).unwrap();
    for (file, bytes) in [
        ("litellm_logo.png", "logo bytes"),
        ("litellm_logo_dark.png", "dark logo bytes"),
        ("litellm_monogram.svg", "monogram bytes"),
        ("litellm_monogram_dark.svg", "dark monogram bytes"),
    ] {
        std::fs::write(export.join("assets/logos").join(file), bytes).unwrap();
    }
    std::fs::write(export.join("favicon.ico"), "icon bytes").unwrap();
    std::fs::write(export.join("_next/static/app.js"), "window.app = true;").unwrap();
    App {
        router: litellm_gateway_ui::dashboard_assets(export),
        _directory: directory,
    }
}

#[rstest]
#[case::dashboard("/ui/", "dashboard", "text/html")]
#[case::login("/ui/login/", "login page", "text/html")]
#[case::javascript("/ui/assets/app.js", "window.app = true;", "text/javascript")]
#[case::stylesheet("/ui/styles.css", "body { color: black; }", "text/css")]
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
#[case::no_implicit_root_assets("/assets/app.js")]
#[case::no_implicit_dashboard_branding("/get_logo_url")]
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
#[case::dashboard("/ui/", "dashboard", "text/html")]
#[case::asset_alias(
    "/litellm-asset-prefix/_next/static/app.js",
    "window.app = true;",
    "text/javascript"
)]
#[case::root_assets("/_next/static/app.js", "window.app = true;", "text/javascript")]
#[case::nested_assets("/ui/_next/static/app.js", "window.app = true;", "text/javascript")]
#[case::logo("/get_image", "logo bytes", "image/png")]
#[case::logo_light("/get_image?theme=light", "logo bytes", "image/png")]
#[case::logo_dark("/get_image?theme=dark", "dark logo bytes", "image/png")]
#[case::monogram("/get_image?variant=monogram", "monogram bytes", "image/svg+xml")]
#[case::monogram_dark(
    "/get_image?theme=dark&variant=monogram",
    "dark monogram bytes",
    "image/svg+xml"
)]
#[case::favicon("/get_favicon", "icon bytes", "image/x-icon")]
#[tokio::test]
async fn dashboard_adapter_preserves_existing_urls(
    dashboard: App,
    #[case] path: &str,
    #[case] expected: &str,
    #[case] mime: &str,
) {
    let response = dashboard
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
#[tokio::test]
async fn logo_discovery_points_to_served_image(dashboard: App) {
    let response = dashboard
        .router
        .clone()
        .oneshot(Request::get("/get_logo_url").body(Body::empty()).unwrap())
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let config: Value =
        serde_json::from_slice(&to_bytes(response.into_body(), 65536).await.unwrap()).unwrap();
    let response = dashboard
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
#[case::logo("/get_image")]
#[case::logo_dark("/get_image?theme=dark")]
#[case::monogram("/get_image?variant=monogram")]
#[case::monogram_dark("/get_image?theme=dark&variant=monogram")]
#[tokio::test]
async fn dashboard_source_serves_every_logo(#[case] path: &str) {
    let export = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../../ui/litellm-dashboard/public");
    let response = litellm_gateway_ui::dashboard_assets(export)
        .oneshot(Request::get(path).body(Body::empty()).unwrap())
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
}
