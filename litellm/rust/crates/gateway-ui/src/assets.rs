use std::path::Path;

use axum::{
    Router,
    extract::{OriginalUri, Request},
    http::Uri,
    middleware::{self, Next},
    response::{IntoResponse, Redirect, Response},
    routing::get,
};
use tower_http::services::ServeDir;

pub fn static_assets(directory: impl AsRef<Path>) -> Router {
    let pages = Router::new()
        .fallback_service(ServeDir::new(directory))
        .layer(middleware::from_fn(restore_redirect));
    Router::new()
        .route(
            "/ui",
            get(|OriginalUri(uri): OriginalUri| async move { append_slash(&uri) }),
        )
        .nest("/ui/", pages)
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
