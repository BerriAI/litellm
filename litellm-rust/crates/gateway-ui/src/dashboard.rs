use std::path::Path;

use axum::{
    Router,
    extract::{Query, Request},
    routing::get,
};
use serde::{Deserialize, Serialize};
use tower::ServiceExt;
use tower_http::services::{ServeDir, ServeFile};

#[derive(Serialize)]
struct Logo {
    logo_url: &'static str,
}

#[derive(Clone, Copy, Deserialize)]
#[serde(rename_all = "lowercase")]
enum Theme {
    Light,
    Dark,
}

#[derive(Clone, Copy, Deserialize)]
#[serde(rename_all = "lowercase")]
enum Variant {
    Full,
    Monogram,
}

#[derive(Deserialize)]
struct LogoQuery {
    theme: Option<Theme>,
    variant: Option<Variant>,
}

fn logo_file(query: &LogoQuery) -> &'static str {
    match (query.variant, query.theme) {
        (Some(Variant::Monogram), Some(Theme::Dark)) => "assets/logos/litellm_monogram_dark.svg",
        (Some(Variant::Monogram), _) => "assets/logos/litellm_monogram.svg",
        (_, Some(Theme::Dark)) => "assets/logos/litellm_logo_dark.png",
        _ => "assets/logos/litellm_logo.png",
    }
}

pub fn dashboard_assets(directory: impl AsRef<Path>) -> Router {
    let directory = directory.as_ref();
    let assets = ServeDir::new(directory.join("_next")).append_index_html_on_directories(false);
    let logos = directory.to_path_buf();

    crate::static_assets(directory)
        .route(
            "/get_logo_url",
            get(|| async {
                axum::Json(Logo {
                    logo_url: "/get_image",
                })
            }),
        )
        .route(
            "/get_image",
            get(move |Query(query): Query<LogoQuery>, request: Request| {
                ServeFile::new(logos.join(logo_file(&query))).oneshot(request)
            }),
        )
        .route_service(
            "/get_favicon",
            ServeFile::new(directory.join("favicon.ico")),
        )
        .nest_service("/_next", assets.clone())
        .nest_service("/litellm-asset-prefix/_next", assets)
}
