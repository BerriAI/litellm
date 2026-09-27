use std::path::Path;

use axum::{
    Router,
    extract::{OriginalUri, Request},
    http::Uri,
    middleware::{self, Next},
    response::{IntoResponse, Redirect, Response},
    routing::get,
};
use serde::Serialize;
use tower_http::services::{ServeDir, ServeFile};

#[derive(Serialize)]
struct Logo {
    logo_url: &'static str,
}

pub fn router(directory: &Path) -> Router {
    let pages = Router::new()
        .fallback_service(ServeDir::new(directory))
        .layer(middleware::from_fn(restore_redirect));
    let assets = ServeDir::new(directory.join("_next")).append_index_html_on_directories(false);
    Router::new()
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
        .route(
            "/ui",
            get(|OriginalUri(uri): OriginalUri| async move { append_slash(&uri) }),
        )
        .nest("/ui/", pages)
        .nest_service("/_next", assets.clone())
        .nest_service("/litellm-asset-prefix/_next", assets)
}

async fn restore_redirect(OriginalUri(uri): OriginalUri, request: Request, next: Next) -> Response {
    let response = next.run(request).await;
    if response.status().is_redirection() {
        return append_slash(&uri).into_response();
    }
    response
}

fn append_slash(uri: &Uri) -> Redirect {
    let location = match uri.query() {
        Some(query) => format!("{}/?{query}", uri.path()),
        None => format!("{}/", uri.path()),
    };
    Redirect::permanent(&location)
}
