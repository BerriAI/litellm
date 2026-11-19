use std::time::Duration;

use axum::http::{HeaderName, HeaderValue, Method};
use litellm_config::Environment;
use litellm_gateway_inference::ExposedHeader;
use litellm_python_compat::unicode::isspace;
use strum::VariantArray;
use tower_http::cors::{AllowHeaders, AllowOrigin, CorsLayer, ExposeHeaders};

/// Starlette's `ALL_METHODS`, what `allow_methods=["*"]` answers a preflight with.
const ALL_METHODS: [Method; 7] = [
    Method::DELETE,
    Method::GET,
    Method::HEAD,
    Method::OPTIONS,
    Method::PATCH,
    Method::POST,
    Method::PUT,
];
const PREFLIGHT_MAX_AGE: Duration = Duration::from_secs(600);

#[derive(Debug, Default, PartialEq)]
pub struct Settings {
    origins: Origins,
    allow_credentials: bool,
}

#[derive(Debug, Default, PartialEq)]
enum Origins {
    #[default]
    Any,
    List(Vec<HeaderValue>),
}

impl Settings {
    /// `_get_cors_config` in `proxy_server.py`: credentials default to on only for listed origins.
    pub fn from_environment(environment: &Environment) -> Self {
        let origins = Origins::parse(environment.litellm_cors_origins.as_deref());
        let allow_credentials = environment
            .litellm_cors_allow_credentials
            .map_or(origins != Origins::Any, |flag| flag.0);
        Self {
            origins,
            allow_credentials,
        }
    }
}

impl Origins {
    /// Unset or blank is `*`, and Starlette allows every origin once `*` is anywhere in the list.
    /// A value of only separators lists nothing, which allows no origin.
    fn parse(raw: Option<&str>) -> Self {
        let Some(raw) = raw.filter(|raw| !raw.trim_matches(isspace).is_empty()) else {
            return Self::Any;
        };
        let entries: Vec<&str> = raw
            .split(',')
            .map(|item| item.trim_matches(isspace))
            .filter(|entry| !entry.is_empty())
            .collect();
        if entries.contains(&"*") {
            return Self::Any;
        }
        Self::List(
            entries
                .into_iter()
                .filter_map(|entry| HeaderValue::from_str(entry).ok())
                .collect(),
        )
    }
}

/// The `CORSMiddleware` that `proxy_server.py` adds.
pub(super) fn layer(settings: &Settings) -> CorsLayer {
    let allow_origin = match (&settings.origins, settings.allow_credentials) {
        (Origins::Any, false) => AllowOrigin::any(),
        (Origins::Any, true) => AllowOrigin::mirror_request(),
        (Origins::List(origins), _) => AllowOrigin::list(origins.iter().cloned()),
    };
    CorsLayer::new()
        .allow_origin(allow_origin)
        .allow_credentials(settings.allow_credentials)
        .allow_methods(ALL_METHODS)
        .allow_headers(AllowHeaders::mirror_request())
        .expose_headers(ExposeHeaders::list(
            ExposedHeader::VARIANTS
                .iter()
                .copied()
                .map(HeaderName::from),
        ))
        .max_age(PREFLIGHT_MAX_AGE)
}

#[cfg(test)]
mod tests {
    use axum::{
        Router,
        body::Body,
        extract::Request,
        http::{HeaderMap, header},
        routing::post,
    };
    use litellm_config::Flag;
    use rstest::rstest;
    use tower::ServiceExt;

    use super::*;

    fn settings(origins: Option<&str>, allow_credentials: Option<bool>) -> Settings {
        Settings::from_environment(&Environment {
            litellm_cors_origins: origins.map(str::to_owned),
            litellm_cors_allow_credentials: allow_credentials.map(Flag),
            ..Environment::default()
        })
    }

    fn listed(origins: &[&'static str]) -> Origins {
        Origins::List(
            origins
                .iter()
                .copied()
                .map(HeaderValue::from_static)
                .collect(),
        )
    }

    async fn respond(settings: &Settings, request: Request) -> HeaderMap {
        Router::new()
            .route("/v1/messages", post(|| async { "ok" }))
            .layer(layer(settings))
            .oneshot(request)
            .await
            .unwrap()
            .headers()
            .clone()
    }

    fn preflight(origin: &str) -> Request {
        Request::options("/v1/messages")
            .header(header::ORIGIN, origin)
            .header(header::ACCESS_CONTROL_REQUEST_METHOD, "POST")
            .header(
                header::ACCESS_CONTROL_REQUEST_HEADERS,
                "authorization,x-custom",
            )
            .body(Body::empty())
            .unwrap()
    }

    fn simple(origin: &str) -> Request {
        Request::post("/v1/messages")
            .header(header::ORIGIN, origin)
            .body(Body::empty())
            .unwrap()
    }

    #[rstest]
    #[case::unset(None, None, Origins::Any, false)]
    #[case::blank(Some("  "), None, Origins::Any, false)]
    #[case::wildcard_in_a_list(Some("https://a.example, *"), None, Origins::Any, false)]
    #[case::list(
        Some(" https://a.example ,,https://b.example"),
        None,
        listed(&["https://a.example", "https://b.example"]),
        true
    )]
    #[case::only_separators(Some(","), None, listed(&[]), true)]
    #[case::credentials_forced_on(None, Some(true), Origins::Any, true)]
    #[case::credentials_forced_off(
        Some("https://a.example"),
        Some(false),
        listed(&["https://a.example"]),
        false
    )]
    fn settings_follow_get_cors_config(
        #[case] origins: Option<&str>,
        #[case] credentials: Option<bool>,
        #[case] expected_origins: Origins,
        #[case] allow_credentials: bool,
    ) {
        assert_eq!(
            settings(origins, credentials),
            Settings {
                origins: expected_origins,
                allow_credentials,
            }
        );
    }

    #[rstest]
    #[tokio::test]
    async fn a_preflight_lists_every_method_and_mirrors_the_requested_headers() {
        let headers = respond(&Settings::default(), preflight("https://site.example")).await;
        assert_eq!(headers[header::ACCESS_CONTROL_ALLOW_ORIGIN], "*");
        assert_eq!(
            headers[header::ACCESS_CONTROL_ALLOW_METHODS],
            "DELETE,GET,HEAD,OPTIONS,PATCH,POST,PUT"
        );
        assert_eq!(
            headers[header::ACCESS_CONTROL_ALLOW_HEADERS],
            "authorization,x-custom"
        );
        assert_eq!(headers[header::ACCESS_CONTROL_MAX_AGE], "600");
        assert!(!headers.contains_key(header::ACCESS_CONTROL_ALLOW_CREDENTIALS));
    }

    #[rstest]
    #[case::listed("https://a.example", Some("https://a.example"))]
    #[case::unlisted("https://evil.example", None)]
    #[tokio::test]
    async fn listed_origins_are_echoed_with_credentials(
        #[case] origin: &str,
        #[case] expected: Option<&str>,
    ) {
        let settings = settings(Some("https://a.example"), None);
        let headers = respond(&settings, simple(origin)).await;
        assert_eq!(
            headers
                .get(header::ACCESS_CONTROL_ALLOW_ORIGIN)
                .map(|value| value.to_str().unwrap()),
            expected
        );
        assert_eq!(headers[header::ACCESS_CONTROL_ALLOW_CREDENTIALS], "true");
    }

    #[rstest]
    #[tokio::test]
    async fn wildcard_origins_with_credentials_reflect_the_origin() {
        let settings = settings(None, Some(true));
        let headers = respond(&settings, simple("https://site.example")).await;
        assert_eq!(
            headers[header::ACCESS_CONTROL_ALLOW_ORIGIN],
            "https://site.example"
        );
        assert_eq!(headers[header::ACCESS_CONTROL_ALLOW_CREDENTIALS], "true");
    }

    #[rstest]
    #[tokio::test]
    async fn responses_expose_the_ui_headers() {
        let headers = respond(&Settings::default(), simple("https://site.example")).await;
        assert_eq!(
            headers[header::ACCESS_CONTROL_EXPOSE_HEADERS],
            "x-litellm-semantic-filter,x-litellm-semantic-filter-tools,x-litellm-adaptive-router-model,\
             x-litellm-applied-guardrails,x-litellm-guardrail-scan-id,x-litellm-guardrail-scan-metadata,\
             x-litellm-cache-key"
        );
    }
}
