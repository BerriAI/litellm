use std::path::Path;

use axum::{Router, routing::get};
use serde::Serialize;
use tower_http::services::{ServeDir, ServeFile};

#[derive(Serialize)]
struct Logo {
    logo_url: &'static str,
}

pub fn dashboard_assets(directory: impl AsRef<Path>) -> Router {
    let directory = directory.as_ref();
    let assets = ServeDir::new(directory.join("_next")).append_index_html_on_directories(false);

    crate::static_assets(directory)
        .route(
            "/get_logo_url",
            get(|| async {
                axum::Json(Logo {
                    logo_url: "/get_image",
                })
            }),
        )
        .route_service(
            "/get_image",
            ServeFile::new(directory.join("assets/logos/litellm_logo.jpg")),
        )
        .route_service(
            "/get_favicon",
            ServeFile::new(directory.join("favicon.ico")),
        )
        .nest_service("/_next", assets.clone())
        .nest_service("/litellm-asset-prefix/_next", assets)
}
