use axum::http::{HeaderValue, header};
use litellm_config::Environment;
use tower::{
    layer::util::Identity,
    util::{Either, option_layer},
};
use tower_http::set_header::SetResponseHeaderLayer;

#[derive(Debug, Default)]
pub struct Settings {
    hsts: bool,
}

impl Settings {
    pub fn from_environment(environment: &Environment) -> Self {
        Self {
            hsts: environment.litellm_enable_hsts.0,
        }
    }
}

type SetHeader = SetResponseHeaderLayer<HeaderValue>;

/// `SecurityHeadersMiddleware`: each header is only a default, a route's own value wins.
pub(super) fn layer(
    settings: &Settings,
) -> (SetHeader, SetHeader, SetHeader, Either<SetHeader, Identity>) {
    (
        SetResponseHeaderLayer::if_not_present(
            header::X_FRAME_OPTIONS,
            HeaderValue::from_static("DENY"),
        ),
        SetResponseHeaderLayer::if_not_present(
            header::CONTENT_SECURITY_POLICY,
            HeaderValue::from_static("frame-ancestors 'none'"),
        ),
        SetResponseHeaderLayer::if_not_present(
            header::X_CONTENT_TYPE_OPTIONS,
            HeaderValue::from_static("nosniff"),
        ),
        option_layer(settings.hsts.then(|| {
            SetResponseHeaderLayer::if_not_present(
                header::STRICT_TRANSPORT_SECURITY,
                HeaderValue::from_static("max-age=31536000; includeSubDomains"),
            )
        })),
    )
}

#[cfg(test)]
mod tests {
    use axum::{Router, body::Body, extract::Request, routing::get};
    use litellm_config::Flag;
    use rstest::rstest;
    use tower::ServiceExt;

    use super::*;

    async fn headers(
        settings: Settings,
        route_frame_options: Option<&'static str>,
    ) -> axum::http::HeaderMap {
        let app = Router::new()
            .route(
                "/",
                get(move || async move {
                    route_frame_options
                        .map(|value| (header::X_FRAME_OPTIONS, HeaderValue::from_static(value)))
                        .into_iter()
                        .collect::<axum::http::HeaderMap>()
                }),
            )
            .layer(layer(&settings));
        app.oneshot(Request::get("/").body(Body::empty()).unwrap())
            .await
            .unwrap()
            .headers()
            .clone()
    }

    #[rstest]
    #[tokio::test]
    async fn every_response_gets_the_anti_framing_headers() {
        let headers = headers(Settings::default(), None).await;
        assert_eq!(headers[header::X_FRAME_OPTIONS], "DENY");
        assert_eq!(
            headers[header::CONTENT_SECURITY_POLICY],
            "frame-ancestors 'none'"
        );
        assert_eq!(headers[header::X_CONTENT_TYPE_OPTIONS], "nosniff");
        assert!(!headers.contains_key(header::STRICT_TRANSPORT_SECURITY));
    }

    #[rstest]
    #[tokio::test]
    async fn a_route_value_is_never_overridden() {
        let headers = headers(Settings::default(), Some("SAMEORIGIN")).await;
        assert_eq!(
            headers
                .get_all(header::X_FRAME_OPTIONS)
                .iter()
                .collect::<Vec<_>>(),
            ["SAMEORIGIN"]
        );
    }

    #[rstest]
    #[case::enabled(true)]
    #[case::disabled(false)]
    #[tokio::test]
    async fn hsts_is_sent_only_when_the_flag_is_on(#[case] expected: bool) {
        let settings = Settings::from_environment(&Environment {
            litellm_enable_hsts: Flag(expected),
            ..Environment::default()
        });
        let headers = headers(settings, None).await;
        assert_eq!(
            headers
                .get(header::STRICT_TRANSPORT_SECURITY)
                .map(|value| value.to_str().unwrap()),
            expected.then_some("max-age=31536000; includeSubDomains")
        );
    }
}
