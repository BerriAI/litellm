//! Which requests the gateway answers itself in front of the Python proxy.
//!
//! `survey` and `plan` decide from the config, once and without IO, what is left to Python.
//! `request` finds what one request carries. This module is the only part that does IO: it looks
//! up what the config does not say, checks the caller, which reads the database for a virtual
//! key, and reads the body.

mod body;
mod plan;
mod request;
mod survey;

use std::sync::Arc;

use axum::{
    body::Body,
    extract::Request,
    http::{Method, request::Parts},
};
use litellm_config::Config;
use litellm_gateway_auth::Auth;
use litellm_gateway_inference::JsonObject;
use litellm_router_core::registry::DeploymentFilter;
use serde_json::{Map, Value};

use crate::{AppState, Resources};
use plan::{Left, Plan};
use survey::Native;

/// Why the gateway does not serve a request the way the Python proxy would.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error, strum::IntoStaticStr)]
pub enum Refused {
    #[error("the gateway is not running the proxy's configuration")]
    #[strum(serialize = "configuration")]
    Configuration,
    #[error("the proxy's configuration has a setting the gateway does not honor yet")]
    #[strum(serialize = "settings")]
    Settings,
    #[error("the gateway does not serve this route")]
    #[strum(serialize = "route")]
    Route,
    #[error("the gateway cannot check this caller's credential")]
    #[strum(serialize = "credential")]
    Credential,
    #[error("the request carries a header only the proxy reads")]
    #[strum(serialize = "headers")]
    Headers,
    #[error("the request body carries something only the proxy reads")]
    #[strum(serialize = "body")]
    Body,
    #[error("the gateway does not serve this model the way the proxy does")]
    #[strum(serialize = "model")]
    Model,
}

fn logged(left: Left<'_>) -> Refused {
    tracing::debug!(why = %left.why, over = left.over, "python answers a request over this");
    left.why
}

struct Judge {
    auth: Auth,
    plan: Plan,
}

impl Judge {
    async fn head(&self, request: &Parts) -> Result<(), Refused> {
        self.plan.everything().map_err(logged)?;
        if (&request.method, request.uri.path()) != (&Method::POST, "/v1/messages") {
            return Err(Refused::Route);
        }
        self.auth
            .authenticate_parts(request)
            .await
            .map_err(|_| Refused::Credential)?;
        self.plan.carrying(&request::head(request)).map_err(logged)
    }

    /// The body parsed, for the route to take without parsing it again.
    fn body(&self, sent: &[u8]) -> Result<JsonObject, Refused> {
        let fields =
            serde_json::from_slice::<Map<String, Value>>(sent).map_err(|_| Refused::Body)?;
        let asked = request::body(&fields).ok_or(Refused::Body)?;
        self.plan.carrying(&asked.carried).map_err(logged)?;
        self.plan.model(&asked.model).map_err(logged)?;
        Ok(JsonObject(fields))
    }
}

fn refused(request: Parts, body: Body, why: Refused) -> (Box<Request>, Refused) {
    (Box::new(Request::from_parts(request, body)), why)
}

/// Admits the requests the gateway serves the way the Python proxy would, and refuses every other
/// one, giving it back as the client sent it, with why.
#[derive(Clone)]
pub struct Admit(Option<Arc<Judge>>);

impl Admit {
    /// Refuses every request, for a gateway that is not running the config Python runs with.
    pub fn unconfigured() -> Self {
        Self(None)
    }

    /// By the `config` the native app `state` is built from. The config is judged here, once, and
    /// every setting found to leave requests to Python is logged with whose it leaves.
    pub fn new(config: &Config, resources: &Resources, state: &AppState) -> Self {
        let deployed: Vec<_> = state
            .inference
            .router
            .registry()
            .deployments(&DeploymentFilter::default())
            .iter()
            .map(|deployment| {
                (
                    deployment.model_name.clone(),
                    deployment.provider().to_owned(),
                )
            })
            .collect();
        let switches: Vec<_> = survey::switches()
            .filter(|name| resources.environment.get(name).is_some())
            .collect();
        let plan = survey::plan(
            config,
            &Native {
                deployed: &deployed,
                switches: &switches,
            },
        );
        for finding in plan.findings() {
            tracing::info!(
                setting = %finding.setting,
                leaves = %finding.leaves,
                "the native app does not honor a setting of the config python runs with, so python answers what it reaches"
            );
        }
        Self::by(state.auth(config), plan)
    }

    fn by(auth: Auth, plan: Plan) -> Self {
        Self(Some(Arc::new(Judge { auth, plan })))
    }

    /// The body is read only for a request that passes every other check, and parsed once: an
    /// admitted request carries it parsed, as the [`JsonObject`] the route takes. A refused request
    /// keeps the bytes the client sent and carries nothing more.
    pub async fn admit(self, request: Request) -> Result<Request, (Box<Request>, Refused)> {
        let (mut parts, body) = request.into_parts();
        let Some(judge) = self.0 else {
            return Err(refused(parts, body, Refused::Configuration));
        };
        if let Err(why) = judge.head(&parts).await {
            return Err(refused(parts, body, why));
        }
        let (body, whole) = body::read(&parts, body).await;
        match whole.map_or(Err(Refused::Body), |sent| judge.body(&sent)) {
            Ok(parsed) => {
                parts.extensions.insert(parsed);
                Ok(Request::from_parts(parts, body))
            }
            Err(why) => Err(refused(parts, body, why)),
        }
    }
}

#[cfg(test)]
mod tests {
    use litellm_gateway_auth::{
        BearerOrXApiKey,
        keys::{KeyHash, KeyLookup, KeyStatus, LookupFuture},
    };
    use litellm_secret_value::SecretValue;
    use rstest::rstest;

    use super::*;

    const CONFIG: &str = "model_list:\n  - model_name: claude\n    litellm_params:\n      model: anthropic/claude-haiku-5-5\n      api_key: provider-key\n  - model_name: gpt\n    litellm_params:\n      model: openai/gpt-5.5\n      api_key: provider-key\ngeneral_settings:\n  master_key: gateway-key\n";
    const MASTER_KEY: (&str, &str) = ("authorization", "Bearer gateway-key");

    const VIRTUAL_KEY: (&str, &str) = ("authorization", "Bearer sk-virtual");

    /// What the database answers for the hash of [`VIRTUAL_KEY`], and nothing for any other.
    #[derive(Clone, Copy)]
    enum Held {
        Active,
        Revoked,
        Restricted,
        Unreachable,
    }

    impl KeyLookup for Held {
        fn lookup<'a>(&'a self, hash: &'a KeyHash) -> LookupFuture<'a> {
            Box::pin(async move {
                if hash != &KeyHash::from_token(&SecretValue::new("sk-virtual")) {
                    return Ok(None);
                }
                match self {
                    Self::Active => Ok(Some(KeyStatus::Active { expires_at: None })),
                    Self::Revoked => Ok(Some(KeyStatus::Revoked)),
                    Self::Restricted => Err(litellm_gateway_auth::Error::Forbidden),
                    Self::Unreachable => Err(litellm_gateway_auth::Error::Unavailable(
                        litellm_error::Source::new(std::io::Error::other("database down")),
                    )),
                }
            })
        }
    }

    fn judged(settings: &str, auth: impl FnOnce(&Config) -> Auth) -> Admit {
        let config = Config::from_yaml(&format!("{CONFIG}{settings}")).unwrap();
        let deployed = [
            ("claude".to_owned(), "anthropic".to_owned()),
            ("gpt".to_owned(), "openai".to_owned()),
        ];
        let native = Native {
            deployed: &deployed,
            switches: &[],
        };
        Admit::by(
            auth(&config).with_extractor(Arc::new(BearerOrXApiKey)),
            survey::plan(&config, &native),
        )
    }

    fn admit(settings: &str) -> Admit {
        judged(settings, Auth::from_config)
    }

    fn admit_holding(held: Held) -> Admit {
        judged("", |config| Auth::with_keys(config, Arc::new(held)))
    }

    fn messages(headers: &[(&str, &str)], model: &str) -> Request {
        headers
            .iter()
            .fold(Request::post("/v1/messages"), |request, (name, value)| {
                request.header(*name, *value)
            })
            .body(Body::from(format!(
                r#"{{"model":"{model}","max_tokens":8,"messages":[]}}"#
            )))
            .unwrap()
    }

    async fn refusal(admit: Admit, request: Request) -> Option<Refused> {
        admit.admit(request).await.err().map(|(_, why)| why)
    }

    #[rstest]
    #[case::bearer(MASTER_KEY)]
    #[case::x_api_key(("x-api-key", "gateway-key"))]
    #[tokio::test]
    async fn a_master_key_caller_of_a_native_model_is_admitted(#[case] credential: (&str, &str)) {
        assert_eq!(
            refusal(admit(""), messages(&[credential], "claude")).await,
            None
        );
    }

    #[rstest]
    #[tokio::test]
    async fn a_virtual_key_the_native_app_authenticates_is_admitted() {
        assert_eq!(
            refusal(
                admit_holding(Held::Active),
                messages(&[VIRTUAL_KEY], "claude")
            )
            .await,
            None
        );
    }

    #[rstest]
    #[case::revoked(Held::Revoked, VIRTUAL_KEY)]
    #[case::holding_what_the_native_app_does_not_honor(Held::Restricted, VIRTUAL_KEY)]
    #[case::unreadable(Held::Unreachable, VIRTUAL_KEY)]
    #[case::unknown(Held::Active, ("authorization", "Bearer sk-someone-else"))]
    #[tokio::test]
    async fn a_virtual_key_the_native_app_does_not_authenticate_is_pythons(
        #[case] held: Held,
        #[case] credential: (&str, &str),
    ) {
        assert_eq!(
            refusal(admit_holding(held), messages(&[credential], "claude")).await,
            Some(Refused::Credential)
        );
    }

    #[rstest]
    #[case::wrong_key(&[("authorization", "Bearer sk-someone-else")])]
    #[case::no_key(&[])]
    #[tokio::test]
    async fn a_caller_the_gateway_cannot_check_is_pythons(#[case] headers: &[(&str, &str)]) {
        assert_eq!(
            refusal(admit(""), messages(headers, "claude")).await,
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
            .header(MASTER_KEY.0, MASTER_KEY.1)
            .body(Body::empty())
            .unwrap();

        assert_eq!(refusal(admit(""), request).await, Some(Refused::Route));
    }

    #[rstest]
    #[case::a_group_of_a_provider_off_the_list("", &[MASTER_KEY], "gpt", Refused::Model)]
    #[case::a_model_the_config_does_not_name("", &[MASTER_KEY], "nope", Refused::Model)]
    #[case::a_header_only_the_proxy_reads(
        "",
        &[MASTER_KEY, ("x-litellm-tags", "paid")],
        "claude",
        Refused::Headers
    )]
    #[case::a_coding_agent_whose_drop_params_the_proxy_defaults(
        "",
        &[MASTER_KEY, ("user-agent", "claude-cli/2.1.0")],
        "claude",
        Refused::Headers
    )]
    #[case::a_setting_the_native_app_does_not_honor(
        "litellm_settings:\n  callbacks: [prometheus]\n",
        &[MASTER_KEY],
        "claude",
        Refused::Settings
    )]
    #[tokio::test]
    async fn a_request_the_native_app_would_not_answer_the_same_is_pythons(
        #[case] settings: &str,
        #[case] headers: &[(&str, &str)],
        #[case] model: &str,
        #[case] why: Refused,
    ) {
        assert_eq!(
            refusal(admit(settings), messages(headers, model)).await,
            Some(why)
        );
    }

    #[rstest]
    #[tokio::test]
    async fn a_coding_agent_is_admitted_when_the_config_sets_drop_params() {
        let request = messages(&[MASTER_KEY, ("user-agent", "claude-cli/2.1.0")], "claude");

        assert_eq!(
            refusal(admit("litellm_settings:\n  drop_params: true\n"), request).await,
            None
        );
    }

    #[rstest]
    #[tokio::test]
    async fn a_setting_that_leaves_everything_is_named_before_the_route() {
        let request = Request::post("/v1/chat/completions")
            .body(Body::empty())
            .unwrap();

        assert_eq!(
            refusal(admit("litellm_settings:\n  cache: true\n"), request).await,
            Some(Refused::Settings)
        );
    }

    #[rstest]
    #[tokio::test]
    async fn a_gateway_that_read_no_config_refuses_every_request() {
        assert_eq!(
            refusal(Admit::unconfigured(), messages(&[MASTER_KEY], "claude")).await,
            Some(Refused::Configuration)
        );
    }

    #[rstest]
    #[case::over_its_caller("Bearer sk-someone-else", "claude", Refused::Credential)]
    #[case::over_its_body("Bearer gateway-key", "claude\",\"user\":\"end-user", Refused::Body)]
    #[case::over_a_body_that_is_not_json("Bearer gateway-key", "claude\"}}", Refused::Body)]
    #[case::over_its_model("Bearer gateway-key", "gpt", Refused::Model)]
    #[tokio::test]
    async fn a_refused_request_comes_back_as_the_client_sent_it(
        #[case] authorization: &str,
        #[case] model: &str,
        #[case] why: Refused,
    ) {
        let sent = format!(r#"{{"model":"{model}","max_tokens":8,"messages":[]}}"#);
        let request = Request::post("/v1/messages?beta=true")
            .header("authorization", authorization)
            .header("x-litellm-call-id", "abc")
            .body(Body::from(sent.clone()))
            .unwrap();

        let (returned, refused) = admit("").admit(request).await.unwrap_err();

        assert_eq!(
            (
                refused,
                returned.method(),
                returned.uri().to_string().as_str(),
                returned.headers().len(),
                returned.headers()["x-litellm-call-id"].to_str().unwrap(),
                returned.extensions().len(),
            ),
            (why, &Method::POST, "/v1/messages?beta=true", 2, "abc", 0)
        );
        let body = axum::body::to_bytes(returned.into_body(), usize::MAX)
            .await
            .unwrap();
        assert_eq!(body, sent);
    }

    #[rstest]
    #[tokio::test]
    async fn an_admitted_request_keeps_its_body_and_carries_it_parsed() {
        let sent = r#"{"model":"claude","max_tokens":8,"messages":[]}"#;

        let admitted = admit("")
            .admit(messages(&[MASTER_KEY], "claude"))
            .await
            .unwrap();

        let parsed = admitted
            .extensions()
            .get::<JsonObject>()
            .map(|body| &body.0);
        assert_eq!(
            parsed,
            serde_json::from_str::<Map<String, Value>>(sent)
                .ok()
                .as_ref()
        );
        let body = axum::body::to_bytes(admitted.into_body(), usize::MAX)
            .await
            .unwrap();
        assert_eq!(body, sent);
    }
}
