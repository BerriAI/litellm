use std::sync::Arc;

use axum::{extract::Request, http::Uri};
use litellm_config::Environment;
use litellm_python_compat::unicode::isspace;
use tower::util::MapRequestLayer;

#[derive(Debug, Default)]
pub struct Settings {
    root_path: String,
    root_paths: Vec<String>,
}

impl Settings {
    /// `SERVER_ROOT_PATH`, which FastAPI takes as written, and `SERVER_ROOT_PATHS`.
    pub fn from_environment(environment: &Environment) -> Self {
        Self {
            root_path: environment.server_root_path.clone(),
            root_paths: root_paths(environment.server_root_paths.as_deref().unwrap_or_default()),
        }
    }

    /// `PerRequestRootPathMiddleware`: the longest `SERVER_ROOT_PATHS` entry the path sits under
    /// replaces `SERVER_ROOT_PATH` for that request.
    /// The path the routes match for `path`, with its root path taken off.
    pub fn route<'a>(&self, path: &'a str) -> &'a str {
        route_path(path, self.root_path(path))
    }

    fn root_path(&self, path: &str) -> &str {
        self.root_paths
            .iter()
            .map(String::as_str)
            .find(|prefix| route_path(path, prefix) != path)
            .unwrap_or(&self.root_path)
    }
}

/// Starlette's `get_route_path`: the root path comes off only on a whole segment.
fn route_path<'a>(path: &'a str, root_path: &str) -> &'a str {
    match path.strip_prefix(root_path) {
        Some("") => "/",
        Some(rest) if rest.starts_with('/') => rest,
        _ => path,
    }
}

/// Routes match the path the client sent with its root path taken off. Python leaves the path
/// alone and strips while matching, which a router cannot do, so this goes outside the router.
pub fn layer(settings: Settings) -> MapRequestLayer<impl Fn(Request) -> Request + Clone> {
    let settings = Arc::new(settings);
    MapRequestLayer::new(move |request| without_root_path(&settings, request))
}

fn without_root_path(settings: &Settings, mut request: Request) -> Request {
    let path = request.uri().path();
    let route = settings.route(path);
    if route == path {
        return request;
    }
    if let Some(uri) = with_path(request.uri(), route) {
        *request.uri_mut() = uri;
    }
    request
}

fn with_path(uri: &Uri, path: &str) -> Option<Uri> {
    let path_and_query = match uri.query() {
        Some(query) => format!("{path}?{query}"),
        None => path.to_owned(),
    };
    let mut parts = uri.clone().into_parts();
    parts.path_and_query = Some(path_and_query.parse().ok()?);
    Uri::from_parts(parts).ok()
}

/// `normalize_root_paths`.
fn root_paths(raw: &str) -> Vec<String> {
    let mut kept: Vec<String> = raw
        .split(',')
        .map(|item| item.trim_matches(isspace))
        .filter(|entry| !entry.is_empty())
        .filter_map(normalized)
        .collect();
    kept.sort_by(|a, b| b.len().cmp(&a.len()).then_with(|| a.cmp(b)));
    kept.dedup();
    kept
}

fn normalized(entry: &str) -> Option<String> {
    if !entry.starts_with('/') {
        tracing::warn!(
            entry,
            "SERVER_ROOT_PATHS entry does not start with '/' and is ignored"
        );
        return None;
    }
    let prefix = entry.trim_end_matches('/');
    if prefix.is_empty() {
        tracing::warn!(
            entry,
            "SERVER_ROOT_PATHS entry is the bare root and is ignored"
        );
        return None;
    }
    Some(prefix.to_owned())
}

#[cfg(test)]
mod tests {
    use axum::{Router, body::Body, http::StatusCode, routing::get};
    use rstest::rstest;
    use tower::{Layer, ServiceExt};

    use super::*;

    fn settings(root_path: &str, root_paths: Option<&str>) -> Settings {
        Settings::from_environment(&Environment {
            server_root_path: root_path.to_owned(),
            server_root_paths: root_paths.map(str::to_owned),
            ..Environment::default()
        })
    }

    async fn routed(
        root_path: &str,
        root_paths: Option<&str>,
        target: &str,
    ) -> (StatusCode, String) {
        let seen = |uri: Uri| async move { uri.to_string() };
        let app = layer(settings(root_path, root_paths)).layer(
            Router::new()
                .route("/v1/messages", get(seen))
                .route("/", get(seen)),
        );
        let response = app
            .oneshot(Request::get(target).body(Body::empty()).unwrap())
            .await
            .unwrap();
        let status = response.status();
        let body = axum::body::to_bytes(response.into_body(), 1024)
            .await
            .unwrap();
        (status, String::from_utf8(body.to_vec()).unwrap())
    }

    #[rstest]
    #[case::no_root_path("", None, "/v1/messages", "/v1/messages")]
    #[case::scalar("/api", None, "/api/v1/messages", "/v1/messages")]
    #[case::scalar_is_optional_on_a_request("/api", None, "/v1/messages", "/v1/messages")]
    #[case::query_is_kept("/api", None, "/api/v1/messages?beta=true", "/v1/messages?beta=true")]
    #[case::root_path_alone("/api", None, "/api", "/")]
    #[case::one_of_several(
        "",
        Some("/tenant-a,/tenant-b"),
        "/tenant-b/v1/messages",
        "/v1/messages"
    )]
    #[case::spelled_loosely(
        "",
        Some(" /tenant-a/ ,, /tenant-a"),
        "/tenant-a/v1/messages",
        "/v1/messages"
    )]
    #[case::longest_wins("", Some("/a,/a/b"), "/a/b/v1/messages", "/v1/messages")]
    #[case::several_override_the_scalar(
        "/api",
        Some("/api/tenant"),
        "/api/tenant/v1/messages",
        "/v1/messages"
    )]
    #[tokio::test]
    async fn a_request_is_routed_without_its_root_path(
        #[case] root_path: &str,
        #[case] root_paths: Option<&str>,
        #[case] target: &str,
        #[case] route: &str,
    ) {
        assert_eq!(
            routed(root_path, root_paths, target).await,
            (StatusCode::OK, route.to_owned())
        );
    }

    #[rstest]
    #[case::not_a_whole_segment("/api", None, "/apiv1/v1/messages")]
    #[case::scalar_with_a_trailing_slash("/api/", None, "/api/v1/messages")]
    #[case::unknown_prefix("", Some("/tenant-a"), "/tenant-c/v1/messages")]
    #[case::entry_without_a_leading_slash("", Some("tenant-a"), "/tenant-a/v1/messages")]
    #[case::scalar_under_a_matched_entry("/api", Some("/tenant"), "/tenant/api/v1/messages")]
    #[tokio::test]
    async fn a_path_under_no_root_path_is_routed_as_sent(
        #[case] root_path: &str,
        #[case] root_paths: Option<&str>,
        #[case] target: &str,
    ) {
        assert_eq!(
            routed(root_path, root_paths, target).await.0,
            StatusCode::NOT_FOUND
        );
    }
}
