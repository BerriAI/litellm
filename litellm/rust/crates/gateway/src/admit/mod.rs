//! Which requests the gateway answers itself. In front of the Python proxy every other request
//! goes to Python, and standalone answers it 501.
//!
//! [`Admit`] holds what the native app was built from, and each request is put to the three stages
//! of checks in `checks`. This is the only part that does IO: it authenticates the caller before
//! the `CALLER` stage and reads the body with `body` before the `BODY` stage.

mod body;
mod checks;

use std::sync::Arc;

use axum::{body::Body, extract::Request, http::request::Parts};
use litellm_config::{Config, Environment};
use litellm_core_utils::get_llm_provider_logic::LlmProviders;
use litellm_gateway_auth::{Auth, Authenticated, Preauthenticated};
use litellm_gateway_auth_core::{Allowlists, authz::Layer};
use litellm_gateway_inference::{JsonObject, routing};
use litellm_model_catalog::CostMap;
use litellm_router_core::{Router, registry::DeploymentFilter};
use serde_json::{Map, Value};

use crate::{
    AppState, Gateway, StoredConfig,
    callbacks::parallel_request_limiter,
    host::{self, NativeHost, NativeRows},
    middleware::request_allowlist,
};
use checks::Fields;
pub use checks::{Reason, Refusal};

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
    #[error("the gateway has not loaded the config the proxy stores in its database yet")]
    #[strum(serialize = "db_config_not_loaded")]
    DbConfigNotLoaded,
}

/// One deployment the router holds: its model group and its provider, or the provider's name when
/// `LlmProviders` does not list it.
type Deployed = (String, Result<LlmProviders, String>);

/// Admits the requests the gateway serves the way the Python proxy would, and refuses every other
/// one, giving it back as the client sent it, with why.
///
/// It is only right about the native app built from the same config, so a new config needs both
/// built again and replaced together.
pub struct Admit {
    config: Config,
    /// The variables only the environment holds, with the config's `environment_variables` in
    /// front of the process's.
    environment: Environment,
    /// Whether `config` holds the config the proxy stores in its database.
    stored: StoredConfig,
    /// Each deployment the router built from `config`, since only the router knows which provider
    /// serves each one.
    deployed: Vec<Deployed>,
    /// The model group of each deployment whose model the cost map holds as another model's alias,
    /// with the name it holds it under.
    aliased: Vec<(String, Box<str>)>,
    /// The router the native app serves through, which says where a request for a name can be
    /// served and what a call served there asks of its host.
    router: Router,
    /// The native host, which says which of a model group's host steps it answers.
    host: NativeHost,
    /// The proxy's address and route allowlists, read by the rule the Python proxy reads them with.
    allowlists: Allowlists,
    auth: Auth<NativeRows>,
    /// The native app's inference, which tells whether a caller may call the model it asks for.
    /// `None` leaves every caller with a model list to Python.
    inference: Option<Arc<Gateway>>,
}

fn refused(request: Parts, body: Body, reason: Reason) -> (Box<Request>, Reason) {
    tracing::debug!(why = %reason.why, over = %reason.over, "admit refused a request over this");
    (Box::new(Request::from_parts(request, body)), reason)
}

/// A key that counts tokens reserves what the body asks for before the call, and only Python's
/// estimator reads a body that carries media, so that call goes to Python.
fn token_estimate(
    caller: &Authenticated<NativeRows>,
    path: &str,
    fields: &mut Fields,
) -> Option<Reason> {
    let counts_tokens = parallel_request_limiter::Caller::of(caller).counts_tokens();
    (counts_tokens && parallel_request_limiter::python_estimates(fields, path))
        .then(|| Reason::new(Refused::Body, "media_under_a_token_limit"))
}

impl Admit {
    /// By the `config` the native app `state` is built from, with its environment under it as
    /// `load_config` leaves it, and whether `config` holds the config the proxy stores in its
    /// database.
    pub fn new(config: &Config, state: &AppState, stored: StoredConfig) -> Self {
        let deployments = state
            .inference
            .router
            .registry()
            .deployments(&DeploymentFilter::default());
        let cost_map = CostMap::bundled();
        let aliased = deployments
            .iter()
            .filter_map(|deployment| {
                let alias = checks::alias(
                    cost_map.catalog(),
                    deployment.provider(),
                    &deployment.params.model,
                )?;
                Some((deployment.model_name.clone(), alias))
            })
            .collect();
        let deployed = deployments
            .iter()
            .map(|deployment| {
                let provider = deployment.provider();
                (
                    deployment.model_name.clone(),
                    provider
                        .parse::<LlmProviders>()
                        .map_err(|_| provider.to_owned()),
                )
            })
            .collect();
        Self {
            environment: state.variables.clone(),
            stored,
            auth: state.auth(config),
            aliased,
            host: state.inference.host().clone(),
            inference: Some(state.inference.clone()),
            ..Self::built(config, state.inference.router.clone(), deployed)
        }
    }

    #[cfg(test)]
    pub(crate) fn over(config: &Config, deployed: Vec<Deployed>) -> Self {
        Self::built(config, crate::boot::unhosted(config), deployed)
    }

    fn built(config: &Config, router: Router, deployed: Vec<Deployed>) -> Self {
        Self {
            router,
            allowlists: Allowlists::new(&config.general_settings),
            auth: host::auth::auth(config, None),
            config: config.clone(),
            environment: Environment::default(),
            stored: StoredConfig::Unused,
            deployed,
            aliased: Vec::new(),
            host: NativeHost::default(),
            inference: None,
        }
    }

    /// A caller whose model lists do not reach the model it asks for goes to Python, which
    /// answers the refusal itself. The native route asks the same question of the same router,
    /// and the auth core answers both, but only after its pre-call hooks have counted the
    /// request, so this is not a denial the native app may answer.
    fn model_access(&self, caller: &Authenticated<NativeRows>, fields: &Fields) -> Option<Reason> {
        let model = fields.get("model")?.as_str()?;
        let allowed = match &self.inference {
            Some(inference) => routing::allows(inference, caller, model),
            None => host::layers(caller).all(|Layer::Key(models)| models.is_empty()),
        };
        (!allowed).then(|| Reason::new(Refused::Credential, "models"))
    }

    /// The body is read only for a request that passes every other check, and parsed once: an
    /// admitted request carries it parsed, as the [`JsonObject`] the route takes. A refused request
    /// keeps the bytes the client sent. Either carries the authentication `admit` ran, so the
    /// native app never authenticates the request again and a key `admit` refused stays refused.
    /// A request the proxy's allowlists leave out is admitted as sent, before its caller or body is
    /// read, as Python refuses it before it reads either, and `middleware::request_allowlist`
    /// refuses it in the native app.
    pub async fn admit(
        self: Arc<Self>,
        request: Request,
    ) -> Result<Request, (Box<Request>, Reason)> {
        let (mut parts, body) = request.into_parts();
        if let Some(reason) = checks::HEAD.iter().find_map(|check| check(&self, &parts)) {
            return Err(refused(parts, body, reason));
        }
        if request_allowlist::checked(
            &self.allowlists,
            parts.uri.path(),
            &parts.headers,
            &parts.extensions,
        )
        .is_err()
        {
            return Ok(Request::from_parts(parts, body));
        }
        let authenticated = self.auth.authenticate_parts(&parts).await;
        if let Some(reason) = checks::CALLER
            .iter()
            .find_map(|check| check(&self, &authenticated))
        {
            if let Err(error) = authenticated {
                parts
                    .extensions
                    .insert(Preauthenticated::<NativeRows>(Err(Arc::new(error))));
            }
            return Err(refused(parts, body, reason));
        }
        let (body, whole) = body::read(&parts, body).await;
        let Some(mut fields) =
            whole.and_then(|sent| serde_json::from_slice::<Map<String, Value>>(&sent).ok())
        else {
            return Err(refused(
                parts,
                body,
                Reason::new(Refused::Body, "unreadable"),
            ));
        };
        if let Some(reason) = checks::BODY.iter().find_map(|check| check(&self, &fields)) {
            return Err(refused(parts, body, reason));
        }
        let over_its_caller = authenticated.as_ref().ok().and_then(|caller| {
            self.model_access(caller, &fields)
                .or_else(|| token_estimate(caller, parts.uri.path(), &mut fields))
        });
        if let Some(reason) = over_its_caller {
            return Err(refused(parts, body, reason));
        }
        parts.extensions.insert(JsonObject(fields));
        if let Ok(caller) = authenticated {
            parts.extensions.insert(Preauthenticated(Ok(caller)));
        }
        Ok(Request::from_parts(parts, body))
    }
}

#[cfg(test)]
mod tests {
    use axum::http::Method;
    use rstest::rstest;
    use serde_json::json;

    use super::*;
    use crate::host::auth::{AuthStore, keys};

    const CONFIG: &str = "model_list:\n  - model_name: claude\n    litellm_params:\n      model: anthropic/claude-haiku-5-5\n      api_key: provider-key\n  - model_name: gpt\n    litellm_params:\n      model: openai/gpt-5.5\n      api_key: provider-key\ngeneral_settings:\n  master_key: gateway-key\n";
    const MASTER_KEY: (&str, &str) = ("authorization", "Bearer gateway-key");

    fn admit() -> Arc<Admit> {
        let config = Config::from_yaml(CONFIG).unwrap();
        let deployed = vec![
            ("claude".to_owned(), Ok(LlmProviders::Anthropic)),
            ("gpt".to_owned(), Ok(LlmProviders::Openai)),
        ];
        Arc::new(Admit::over(&config, deployed))
    }

    const VIRTUAL_KEY: (&str, &str) = ("authorization", "Bearer sk-virtual");

    /// What the database holds for [`VIRTUAL_KEY`], the only key it holds.
    #[derive(Clone, Copy)]
    enum Stored {
        Active,
        Revoked,
        Owned,
        Unreachable,
    }

    /// The admit over a database storing the key as `stored` says, with that database.
    async fn holding(stored: Stored) -> (Arc<Admit>, Option<keys::Held>) {
        let columns = match stored {
            Stored::Active => Some(json!({})),
            Stored::Revoked => Some(json!({"blocked": true})),
            Stored::Owned => Some(json!({"user_id": "owner"})),
            Stored::Unreachable => None,
        };
        let held = match columns {
            Some(columns) => Some(keys::Held::holding("sk-virtual", columns).await),
            None => None,
        };
        let database = held
            .as_ref()
            .map_or_else(keys::unreachable, keys::Held::database);
        let config = Config::from_yaml(CONFIG).unwrap();
        let admit = Arc::new(Admit {
            auth: host::auth::auth(
                &config,
                Some(Arc::new(AuthStore::new(keys::memory_cache(), database))),
            ),
            ..Admit::over(
                &config,
                vec![("claude".to_owned(), Ok(LlmProviders::Anthropic))],
            )
        });
        (admit, held)
    }

    #[rstest]
    #[case::active(Stored::Active, VIRTUAL_KEY, None)]
    #[case::revoked(Stored::Revoked, VIRTUAL_KEY, Some("invalid_token"))]
    #[case::holding_what_the_native_app_does_not_serve(
        Stored::Owned,
        VIRTUAL_KEY,
        Some("not_native")
    )]
    #[case::unreadable(Stored::Unreachable, VIRTUAL_KEY, Some("unavailable"))]
    #[case::unknown(
        Stored::Active,
        ("authorization", "Bearer sk-someone-else"),
        Some("invalid_token")
    )]
    #[tokio::test(flavor = "multi_thread")]
    async fn a_virtual_key_is_admitted_only_when_the_native_app_authenticates_it(
        #[case] stored: Stored,
        #[case] credential: (&str, &str),
        #[case] over: Option<&str>,
    ) {
        let (admit, _held) = holding(stored).await;

        let refused = admit
            .admit(messages(&[credential], "claude"))
            .await
            .err()
            .map(|(_, reason)| (reason.why, reason.over));
        assert_eq!(refused, over.map(|over| (Refused::Credential, over.into())));
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

    #[rstest]
    #[case::over_its_caller("Bearer sk-someone-else", "claude", Refused::Credential, true)]
    #[case::over_its_body(
        "Bearer gateway-key",
        "claude\",\"user\":\"end-user",
        Refused::Body,
        false
    )]
    #[case::over_a_body_that_is_not_json("Bearer gateway-key", "claude\"}}", Refused::Body, false)]
    #[case::over_its_model("Bearer gateway-key", "gpt", Refused::Model, false)]
    #[tokio::test]
    async fn a_refused_request_comes_back_as_the_client_sent_it_with_the_key_refusal_admit_made(
        #[case] authorization: &str,
        #[case] model: &str,
        #[case] why: Refused,
        #[case] key_refused: bool,
    ) {
        let sent = format!(r#"{{"model":"{model}","max_tokens":8,"messages":[]}}"#);
        let request = Request::post("/v1/messages?beta=true")
            .header("authorization", authorization)
            .header("x-litellm-call-id", "abc")
            .body(Body::from(sent.clone()))
            .unwrap();

        let (returned, refused) = admit().admit(request).await.unwrap_err();

        assert_eq!(
            (
                refused.why,
                returned.method(),
                returned.uri().to_string().as_str(),
                returned.headers().len(),
                returned.headers()["x-litellm-call-id"].to_str().unwrap(),
                returned.extensions().len(),
                matches!(
                    returned.extensions().get::<Preauthenticated<NativeRows>>(),
                    Some(Preauthenticated(Err(_)))
                ),
            ),
            (
                why,
                &Method::POST,
                "/v1/messages?beta=true",
                2,
                "abc",
                usize::from(key_refused),
                key_refused
            )
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

        let admitted = admit()
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
