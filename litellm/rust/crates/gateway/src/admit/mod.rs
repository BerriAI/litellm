//! Which requests the gateway answers itself in front of the Python proxy.

use std::sync::Arc;

use axum::{
    extract::Request,
    http::{Method, request::Parts},
};
use litellm_config::Config;
use litellm_gateway_auth::{Auth, BearerOrXApiKey};

/// Why the gateway does not serve a request the way the Python proxy would.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error, strum::IntoStaticStr)]
pub enum Refused {
    #[error("the gateway is not running the proxy's configuration")]
    #[strum(serialize = "configuration")]
    Configuration,
    #[error("the gateway does not serve this route")]
    #[strum(serialize = "route")]
    Route,
    #[error("the gateway cannot check this caller's credential")]
    #[strum(serialize = "credential")]
    Credential,
    #[error("the Python proxy checks required customer headers")]
    #[strum(serialize = "required_customer_headers")]
    RequiredCustomerHeaders,
}

/// Admits the requests the gateway serves the way the Python proxy would, and refuses every other
/// one, giving it back as the client sent it, with why.
#[derive(Clone)]
pub struct Admit {
    auth: Option<Auth>,
    requires_customer_headers: bool,
}

impl Admit {
    /// Refuses every request, for a gateway that is not running the config Python runs with.
    pub fn unconfigured() -> Self {
        Self {
            auth: None,
            requires_customer_headers: false,
        }
    }

    /// By the `config` the native app is built from.
    pub fn new(config: &Config) -> Self {
        Self {
            auth: Some(Auth::from_config(config).with_extractor(Arc::new(BearerOrXApiKey))),
            requires_customer_headers: !config
                .general_settings
                .required_customer_headers
                .is_empty(),
        }
    }

    pub async fn admit(self, request: Request) -> Result<Request, (Box<Request>, Refused)> {
        let (parts, body) = request.into_parts();
        let checked = self.check(&parts).await;
        let request = Request::from_parts(parts, body);
        match checked {
            Ok(()) => Ok(request),
            Err(why) => Err((Box::new(request), why)),
        }
    }

    async fn check(&self, request: &Parts) -> Result<(), Refused> {
        let auth = self.auth.as_ref().ok_or(Refused::Configuration)?;
        if (&request.method, request.uri.path()) != (&Method::POST, "/v1/messages") {
            return Err(Refused::Route);
        }
        auth.authenticate_parts(request)
            .await
            .map_err(|_| Refused::Credential)?;
        if self.requires_customer_headers {
            return Err(Refused::RequiredCustomerHeaders);
        }
        // TODO: judge the config's settings and the request's headers and body before admitting, since a setting or parameter only Python honors is served natively until then
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use axum::body::Body;
    use rstest::{fixture, rstest};

    use super::*;

    #[fixture]
    fn admit(#[default("[]")] required: &str) -> Admit {
        Admit::new(
            &Config::from_yaml(&format!(
                "model_list: []\ngeneral_settings:\n  master_key: gateway-key\n  required_customer_headers: {required}\n"
            ))
            .unwrap(),
        )
    }

    fn messages(header: Option<(&str, &str)>) -> Request {
        header
            .into_iter()
            .fold(Request::post("/v1/messages"), |request, (name, value)| {
                request.header(name, value)
            })
            .header("x-tenant-id", "tenant")
            .header("x-user-id", "member")
            .body(Body::empty())
            .unwrap()
    }

    async fn refusal(admit: Admit, request: Request) -> Option<Refused> {
        admit.admit(request).await.err().map(|(_, why)| why)
    }

    #[rstest]
    #[case::bearer(("authorization", "Bearer gateway-key"), "[]", None)]
    #[case::x_api_key(("x-api-key", "gateway-key"), "[]", None)]
    #[case::one_header(("authorization", "Bearer gateway-key"), "[X-Tenant-ID]", Some(Refused::RequiredCustomerHeaders))]
    #[case::multiple_headers(("x-api-key", "gateway-key"), "[X-Tenant-ID, X-User-ID]", Some(Refused::RequiredCustomerHeaders))]
    #[tokio::test]
    async fn a_master_key_caller_of_messages_is_deferred_when_customer_headers_are_required(
        #[case] credential: (&str, &str),
        #[case] _required: &str,
        #[case] expected: Option<Refused>,
        #[with(_required)] admit: Admit,
    ) {
        assert_eq!(refusal(admit, messages(Some(credential))).await, expected);
    }

    #[rstest]
    #[case::wrong_key(Some(("authorization", "Bearer sk-someone-else")))]
    #[case::no_key(None)]
    #[tokio::test]
    async fn a_caller_the_gateway_cannot_check_is_pythons(
        admit: Admit,
        #[case] credential: Option<(&str, &str)>,
    ) {
        assert_eq!(
            refusal(admit, messages(credential)).await,
            Some(Refused::Credential)
        );
    }

    #[rstest]
    #[case::another_path(Method::POST, "/v1/chat/completions")]
    #[case::another_method(Method::GET, "/v1/messages")]
    #[case::a_longer_path(Method::POST, "/v1/messages/count_tokens")]
    #[tokio::test]
    async fn a_route_the_gateway_does_not_serve_is_pythons(
        admit: Admit,
        #[case] method: Method,
        #[case] path: &str,
    ) {
        let request = Request::builder()
            .method(method)
            .uri(path)
            .header("authorization", "Bearer gateway-key")
            .body(Body::empty())
            .unwrap();

        assert_eq!(refusal(admit, request).await, Some(Refused::Route));
    }

    #[rstest]
    #[tokio::test]
    async fn a_gateway_that_read_no_config_refuses_every_request() {
        let request = messages(Some(("authorization", "Bearer gateway-key")));

        assert_eq!(
            refusal(Admit::unconfigured(), request).await,
            Some(Refused::Configuration)
        );
    }

    #[rstest]
    #[tokio::test]
    async fn a_refused_request_comes_back_as_the_client_sent_it(admit: Admit) {
        let request = Request::post("/v1/messages?beta=true")
            .header("authorization", "Bearer sk-someone-else")
            .header("x-litellm-call-id", "abc")
            .body(Body::from("payload"))
            .unwrap();

        let (returned, _) = admit.admit(request).await.unwrap_err();

        assert_eq!(
            (
                returned.method(),
                returned.uri().to_string().as_str(),
                returned.headers().len(),
                returned.headers()["x-litellm-call-id"].to_str().unwrap(),
            ),
            (&Method::POST, "/v1/messages?beta=true", 2, "abc")
        );
        let body = axum::body::to_bytes(returned.into_body(), usize::MAX)
            .await
            .unwrap();
        assert_eq!(body, "payload");
    }
}
