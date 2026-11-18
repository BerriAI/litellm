//! Which requests the gateway answers itself in front of the Python proxy.

mod body;
mod settings;

use std::sync::Arc;

use axum::{
    extract::Request,
    http::{Method, request::Parts},
};
use litellm_config::Config;
use litellm_gateway_auth::{Auth, BearerOrXApiKey};

use crate::{AppState, Resources};

/// Why the gateway does not serve a request the way the Python proxy would.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error, strum::IntoStaticStr)]
pub enum Refused {
    #[error("the gateway is not running the proxy's configuration")]
    #[strum(serialize = "configuration")]
    Configuration,
    #[error("the proxy's configuration has a setting the gateway does not honor yet")]
    #[strum(serialize = "settings")]
    Settings,
    #[error("the proxy has a database, which the gateway does not read yet")]
    #[strum(serialize = "database")]
    Database,
    #[error("the gateway does not serve this route")]
    #[strum(serialize = "route")]
    Route,
    #[error("the gateway cannot check this caller's credential")]
    #[strum(serialize = "credential")]
    Credential,
    #[error("the request carries a LiteLLM header the gateway does not read")]
    #[strum(serialize = "headers")]
    Headers,
    #[error("the request body carries something only the proxy reads")]
    #[strum(serialize = "body")]
    Body,
}

/// The one `x-litellm-*` request header the native app reads as the proxy does. The proxy reads
/// a request's model, tags, timeouts, retries and more from the others.
const CALL_ID: &str = "x-litellm-call-id";

/// Whether `request` carries an `x-litellm-*` header only the proxy reads.
fn has_proxy_only_header(request: &Parts) -> bool {
    request
        .headers
        .keys()
        .map(|name| name.as_str())
        .any(|name| name.starts_with("x-litellm-") && name != CALL_ID)
}

/// Admits the requests the gateway serves the way the Python proxy would, and refuses every other
/// one, giving it back as the client sent it, with why.
#[derive(Clone)]
pub struct Admit(Result<Auth, Refused>);

impl Admit {
    /// Refuses every request, for a gateway that is not running the config Python runs with.
    pub fn unconfigured() -> Self {
        Self(Err(Refused::Configuration))
    }

    /// By the `config` the native app `state` is built from. The config is judged here, once,
    /// since what it sets applies to every request: one setting the native app does not honor,
    /// or a database, which Python reads keys from and can reload its config from, leaves every
    /// request to Python.
    pub async fn new(config: &Config, resources: &Resources, state: &AppState) -> Self {
        let unhonored =
            settings::unhonored(config).or_else(|| settings::unserved(&state.inference.router));
        if let Some(setting) = unhonored {
            tracing::info!(
                %setting,
                "the native app does not honor a setting of the config python runs with, so python answers every request"
            );
            return Self(Err(Refused::Settings));
        }
        if has_database(config, resources).await {
            tracing::info!(
                "python runs with a database the native app does not read yet, so python answers every request"
            );
            return Self(Err(Refused::Database));
        }
        Self(Ok(
            Auth::from_config(config).with_extractor(Arc::new(BearerOrXApiKey))
        ))
    }

    /// The body is read only for a request that passes every other check, and a refused request
    /// keeps the bytes the client sent.
    pub async fn admit(self, request: Request) -> Result<Request, (Box<Request>, Refused)> {
        let (parts, body) = request.into_parts();
        if let Err(why) = self.check(&parts).await {
            return Err((Box::new(Request::from_parts(parts, body)), why));
        }
        let (body, honored) = body::judged(&parts, body).await;
        let request = Request::from_parts(parts, body);
        match honored {
            true => Ok(request),
            false => Err((Box::new(request), Refused::Body)),
        }
    }

    async fn check(&self, request: &Parts) -> Result<(), Refused> {
        let auth = self.0.as_ref().map_err(|why| *why)?;
        if (&request.method, request.uri.path()) != (&Method::POST, "/v1/messages") {
            return Err(Refused::Route);
        }
        auth.authenticate_parts(request)
            .await
            .map_err(|_| Refused::Credential)?;
        if has_proxy_only_header(request) {
            return Err(Refused::Headers);
        }
        Ok(())
    }
}

/// Whether Python runs with a database, including one the gateway opens no pool for.
async fn has_database(config: &Config, resources: &Resources) -> bool {
    !matches!(
        litellm_db_resolve::resolve(&config.general_settings, &resources.secrets).await,
        Ok(None)
    )
}

#[cfg(test)]
mod tests {
    use axum::body::Body;
    use rstest::rstest;

    use super::*;

    const ANTHROPIC: &str =
        "  - model_name: claude\n    litellm_params:\n      model: anthropic/claude-haiku-5-5\n";
    const OPENAI: &str = "  - model_name: gpt\n    litellm_params:\n      model: openai/gpt-5.5\n";

    fn no_variables(_: &str) -> Option<String> {
        None
    }

    async fn admit_over(
        yaml: &str,
        environment: impl Fn(&str) -> Option<String> + Send + Sync + 'static,
    ) -> Admit {
        crate::install_crypto_provider();
        let config = Config::from_yaml(yaml).unwrap();
        let resources = Resources::over(&config, Arc::new(environment)).unwrap();
        let state = AppState::over(&config, &resources).await.unwrap();
        Admit::new(&config, &resources, &state).await
    }

    async fn admit(settings: &str) -> Admit {
        admit_over(
            &format!(
                "model_list:\n{ANTHROPIC}general_settings:\n  master_key: gateway-key\n{settings}"
            ),
            no_variables,
        )
        .await
    }

    const BODY: &str =
        r#"{"model":"claude","max_tokens":16,"messages":[{"role":"user","content":"hi"}]}"#;

    fn messages(header: Option<(&str, &str)>) -> Request {
        header
            .into_iter()
            .fold(Request::post("/v1/messages"), |request, (name, value)| {
                request.header(name, value)
            })
            .body(Body::from(BODY))
            .unwrap()
    }

    fn by_the_master_key() -> Request {
        messages(Some(("authorization", "Bearer gateway-key")))
    }

    async fn refusal(admit: Admit, request: Request) -> Option<Refused> {
        admit.admit(request).await.err().map(|(_, why)| why)
    }

    #[rstest]
    #[case::bearer(("authorization", "Bearer gateway-key"))]
    #[case::x_api_key(("x-api-key", "gateway-key"))]
    #[tokio::test]
    async fn a_master_key_caller_of_messages_is_admitted(#[case] credential: (&str, &str)) {
        assert_eq!(
            refusal(admit("").await, messages(Some(credential))).await,
            None
        );
    }

    #[rstest]
    #[case::wrong_key(Some(("authorization", "Bearer sk-someone-else")))]
    #[case::no_key(None)]
    #[tokio::test]
    async fn a_caller_the_gateway_cannot_check_is_pythons(
        #[case] credential: Option<(&str, &str)>,
    ) {
        assert_eq!(
            refusal(admit("").await, messages(credential)).await,
            Some(Refused::Credential)
        );
    }

    #[rstest]
    #[case::another_path(Method::POST, "/v1/chat/completions")]
    #[case::another_method(Method::GET, "/v1/messages")]
    #[case::a_longer_path(Method::POST, "/v1/messages/count_tokens")]
    #[tokio::test]
    async fn a_route_the_gateway_does_not_serve_is_pythons(
        #[case] method: Method,
        #[case] path: &str,
    ) {
        let request = Request::builder()
            .method(method)
            .uri(path)
            .header("authorization", "Bearer gateway-key")
            .body(Body::empty())
            .unwrap();

        assert_eq!(
            refusal(admit("").await, request).await,
            Some(Refused::Route)
        );
    }

    #[rstest]
    #[case::a_timeout("x-litellm-timeout", "30", Some(Refused::Headers))]
    #[case::tags("x-litellm-tags", "team-a", Some(Refused::Headers))]
    #[case::the_call_id_the_native_app_reads("x-litellm-call-id", "abc", None)]
    #[case::a_provider_header("anthropic-beta", "feature", None)]
    #[tokio::test]
    async fn a_litellm_header_only_python_reads_leaves_the_request_to_python(
        #[case] name: &'static str,
        #[case] value: &'static str,
        #[case] expected: Option<Refused>,
    ) {
        let request = Request::post("/v1/messages")
            .header("authorization", "Bearer gateway-key")
            .header(name, value)
            .body(Body::from(BODY))
            .unwrap();

        assert_eq!(refusal(admit("").await, request).await, expected);
    }

    #[rstest]
    #[tokio::test]
    async fn a_body_with_a_param_only_python_reads_comes_back_as_the_client_sent_it() {
        let sent = r#"{"model":"claude","max_tokens":16,"messages":[],"user":"end-user"}"#;
        let request = Request::post("/v1/messages")
            .header("authorization", "Bearer gateway-key")
            .body(Body::from(sent))
            .unwrap();

        let (returned, why) = admit("").await.admit(request).await.unwrap_err();

        let body = axum::body::to_bytes(returned.into_body(), usize::MAX)
            .await
            .unwrap();
        assert_eq!((why, body), (Refused::Body, axum::body::Bytes::from(sent)));
    }

    #[rstest]
    #[case::a_setting_only_python_honors("litellm_settings:\n  callbacks: [prometheus]\n")]
    #[case::a_router_setting_only_python_honors("router_settings:\n  redis_host: localhost\n")]
    #[tokio::test]
    async fn a_config_the_native_app_does_not_honor_leaves_every_request_to_python(
        #[case] settings: &str,
    ) {
        assert_eq!(
            refusal(admit(settings).await, by_the_master_key()).await,
            Some(Refused::Settings)
        );
    }

    #[rstest]
    #[tokio::test]
    async fn a_deployment_native_messages_does_not_serve_leaves_every_request_to_python() {
        let admit = admit_over(
            &format!(
                "model_list:\n{ANTHROPIC}{OPENAI}general_settings:\n  master_key: gateway-key\n"
            ),
            no_variables,
        )
        .await;

        assert_eq!(
            refusal(admit, by_the_master_key()).await,
            Some(Refused::Settings)
        );
    }

    #[rstest]
    #[case::in_the_config(
        "  database_url: postgresql://user:pass@127.0.0.1:5432/litellm\n",
        None
    )]
    #[case::in_the_environment(
        "",
        Some(("DATABASE_URL", "postgresql://user:pass@127.0.0.1:5432/litellm"))
    )]
    #[case::behind_token_auth_the_gateway_opens_no_pool_for("", Some(("IAM_TOKEN_DB_AUTH", "true")))]
    #[tokio::test]
    async fn a_proxy_with_a_database_leaves_every_request_to_python(
        #[case] general: &str,
        #[case] variable: Option<(&'static str, &'static str)>,
    ) {
        let admit = admit_over(
            &format!(
                "model_list:\n{ANTHROPIC}general_settings:\n  master_key: gateway-key\n{general}"
            ),
            move |name: &str| {
                variable
                    .filter(|(set, _)| *set == name)
                    .map(|(_, value)| value.to_owned())
            },
        )
        .await;

        assert_eq!(
            refusal(admit, by_the_master_key()).await,
            Some(Refused::Database)
        );
    }

    #[rstest]
    #[tokio::test]
    async fn a_gateway_that_runs_no_config_refuses_every_request() {
        assert_eq!(
            refusal(Admit::unconfigured(), by_the_master_key()).await,
            Some(Refused::Configuration)
        );
    }

    #[rstest]
    #[tokio::test]
    async fn a_refused_request_comes_back_as_the_client_sent_it() {
        let request = Request::post("/v1/messages?beta=true")
            .header("authorization", "Bearer sk-someone-else")
            .header("x-litellm-call-id", "abc")
            .body(Body::from("payload"))
            .unwrap();

        let (returned, _) = admit("").await.admit(request).await.unwrap_err();

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
