//! Which requests the gateway answers itself. In front of the Python proxy every other request
//! goes to Python, and standalone answers it 501.
//!
//! [`Admit`] holds what the native app was built from, and each request is put to the three stages
//! of checks in `checks`. This is the only part that does IO: it authenticates the caller before
//! the `CALLER` stage, reads the body with `body` before the `BODY` stage, and reads the end user
//! the body names once every check has passed.

mod body;
mod checks;
mod groups;

use std::sync::Arc;

use axum::{body::Body, extract::Request, http::request::Parts};
use litellm_config::{Config, Environment};
use litellm_core_utils::get_llm_provider_logic::LlmProviders;
use litellm_gateway_auth::{Auth, BearerOrXApiKey, Permissions, Preauthenticated};
use litellm_gateway_inference::{JsonObject, routing};
use litellm_model_catalog::CostMap;
use litellm_router_core::registry::DeploymentFilter;
use serde_json::{Map, Value};

use crate::{
    AppState, EndUserLookup, Gateway,
    callbacks::parallel_request_limiter::{self, KeyLimiter},
    end_users,
};
pub use checks::Reason;
use checks::{Authenticated, Fields};
use groups::Groups;

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
    /// The variables only the environment holds, as `main` read them.
    environment: Environment,
    /// Whether the proxy has a database, where it looks up the end user a request names.
    database: bool,
    /// The end users of that database. `None` when the native app built no pool for it, which
    /// leaves every body that names an end user to Python.
    end_users: Option<Arc<dyn EndUserLookup>>,
    /// The variable that has the proxy serve models and settings from its database on top of the
    /// config. `general_settings.store_model_in_db` is a setting of the config itself.
    stored_models: Option<Box<str>>,
    /// Each deployment the router built from `config`, since only the router knows which provider
    /// serves each one.
    deployed: Vec<Deployed>,
    /// The model group of each deployment whose model the cost map holds as another model's alias,
    /// with the name it holds it under.
    aliased: Vec<(String, Box<str>)>,
    /// `None` when a fallback list or an alias is written in a shape that is not read.
    groups: Option<Groups>,
    auth: Auth,
    /// The native app's inference, which tells whether a caller may call the model it asks for.
    /// `None` leaves every caller with a model list to Python.
    inference: Option<Arc<Gateway>>,
    /// The native app's limiter, which tells whether a caller's call to a model counts tokens.
    /// `None` only knows the limits of the caller's key.
    key_limiter: Option<Arc<KeyLimiter>>,
}

/// The variables the proxy builds a database from when `database_url` names none.
const DATABASE_VARIABLES: [&str; 4] = [
    "DATABASE_URL",
    "DATABASE_HOST",
    "IAM_TOKEN_DB_AUTH",
    "AZURE_POSTGRESQL_AUTH",
];

const STORE_MODEL_IN_DB: &str = "STORE_MODEL_IN_DB";

fn refused(request: Parts, body: Body, reason: Reason) -> (Box<Request>, Reason) {
    tracing::debug!(why = %reason.why, over = %reason.over, "admit refused a request over this");
    (Box::new(Request::from_parts(request, body)), reason)
}

impl Admit {
    /// By the `config` the native app `state` is built from, with its environment under it as
    /// `load_config` leaves it, and the `environment` `main` read.
    pub fn new(config: &Config, environment: &Environment, state: &AppState) -> Self {
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
            environment: environment.clone(),
            database: config.general_settings.database_url.is_some()
                || DATABASE_VARIABLES
                    .into_iter()
                    .any(|variable| state.environment.get(variable).is_some()),
            stored_models: state
                .environment
                .get(STORE_MODEL_IN_DB)
                .map(|_| STORE_MODEL_IN_DB.into()),
            end_users: state.end_users.clone(),
            auth: state.auth(config),
            aliased,
            inference: Some(state.inference.clone()),
            key_limiter: Some(state.key_limiter.clone()),
            ..Self::over(config, deployed)
        }
    }

    fn over(config: &Config, deployed: Vec<Deployed>) -> Self {
        Self {
            groups: Groups::read(
                config
                    .model_list
                    .iter()
                    .map(|model| model.model_name.as_str()),
                &config.router_settings,
            ),
            auth: Auth::from_config(config).with_extractor(Arc::new(BearerOrXApiKey)),
            config: config.clone(),
            environment: Environment::default(),
            database: false,
            end_users: None,
            stored_models: None,
            deployed,
            aliased: Vec::new(),
            inference: None,
            key_limiter: None,
        }
    }

    /// A caller whose model list does not let it call the model it asks for goes to Python, which
    /// answers the refusal itself. The native route asks the same question of the same router.
    async fn model_access(&self, authenticated: &Authenticated, fields: &Fields) -> Option<Reason> {
        let identity = authenticated.as_ref().ok()?;
        let Permissions::Models(_) = identity.caller().restrictions() else {
            return None;
        };
        let model = fields.get("model")?.as_str()?;
        let allowed = match &self.inference {
            Some(inference) => routing::allows(inference, identity, model).await,
            None => false,
        };
        (!allowed).then(|| Reason::new(Refused::Credential, "models"))
    }

    /// A caller under a token limit, its key's or the default one of the model's deployments,
    /// reserves what the body asks for before the call, and only Python's estimator reads a body
    /// that carries media, so that call goes to Python.
    fn token_estimate(
        &self,
        authenticated: &Authenticated,
        path: &str,
        fields: &mut Fields,
    ) -> Option<Reason> {
        let identity = authenticated.as_ref().ok()?;
        let model = fields.get("model").and_then(Value::as_str);
        let counts_tokens = match &self.key_limiter {
            Some(limiter) => limiter.counts_tokens(identity, model),
            None => identity.caller().authentication().limits.counts_tokens(),
        };
        (counts_tokens && parallel_request_limiter::python_estimates(fields, path))
            .then(|| Reason::new(Refused::Body, "media_under_a_token_limit"))
    }

    /// With a database the proxy applies the end user a body names to any caller: its block, its
    /// budget, its limits, its region and its models. The native app applies none of them, so that
    /// body is only kept when the end user's row holds none, or there is no row.
    async fn end_user(&self, fields: &Fields) -> Option<Reason> {
        let end_user_id = checks::end_user(fields).filter(|_| self.database)?;
        let held = match &self.end_users {
            Some(end_users) => end_users::unhonored(end_users.as_ref(), end_user_id).await?,
            None => end_users::UNREAD,
        };
        tracing::debug!(held, "an end user holds what the native app does not honor");
        Some(Reason::new(Refused::Body, "end_user"))
    }

    /// The body is read only for a request that passes every other check, and parsed once: an
    /// admitted request carries it parsed, as the [`JsonObject`] the route takes. A refused request
    /// keeps the bytes the client sent and carries nothing more.
    pub async fn admit(
        self: Arc<Self>,
        request: Request,
    ) -> Result<Request, (Box<Request>, Reason)> {
        let (mut parts, body) = request.into_parts();
        if let Some(reason) = checks::HEAD.iter().find_map(|check| check(&self, &parts)) {
            return Err(refused(parts, body, reason));
        }
        let authenticated = self.auth.authenticate_parts(&parts).await;
        if let Some(reason) = checks::CALLER
            .iter()
            .find_map(|check| check(&self, &authenticated))
        {
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
        if let Some(reason) = self.model_access(&authenticated, &fields).await {
            return Err(refused(parts, body, reason));
        }
        if let Some(reason) = self.token_estimate(&authenticated, parts.uri.path(), &mut fields) {
            return Err(refused(parts, body, reason));
        }
        if let Some(reason) = self.end_user(&fields).await {
            return Err(refused(parts, body, reason));
        }
        parts.extensions.insert(JsonObject(fields));
        if let Ok(identity) = authenticated {
            parts.extensions.insert(Preauthenticated(identity));
        }
        Ok(Request::from_parts(parts, body))
    }
}

#[cfg(test)]
mod tests {
    use axum::http::Method;
    use litellm_repositories::KeyView;
    use rstest::rstest;

    use super::*;
    use crate::host::auth::{KeyRows, RowFuture, authenticator, tests::plain};

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

    /// What the database answers for the hash of [`VIRTUAL_KEY`], and nothing for any other.
    #[derive(Clone, Copy)]
    enum Held {
        Active,
        Revoked,
        Owned,
        Unreachable,
    }

    impl KeyRows for Held {
        fn find<'a>(&'a self, hash: &'a str) -> RowFuture<'a> {
            Box::pin(async move {
                let key = plain("sk-virtual");
                if hash != key.token {
                    return Ok(None);
                }
                match self {
                    Self::Active => Ok(Some(key)),
                    Self::Revoked => Ok(Some(KeyView {
                        blocked: Some(true),
                        ..key
                    })),
                    Self::Owned => Ok(Some(KeyView {
                        user_id: Some("owner".to_owned()),
                        ..key
                    })),
                    Self::Unreachable => Err(litellm_error::Source::new(std::io::Error::other(
                        "database down",
                    ))),
                }
            })
        }
    }

    fn holding(held: Held) -> Arc<Admit> {
        let config = Config::from_yaml(CONFIG).unwrap();
        Arc::new(Admit {
            auth: Auth::with_keys(&config, authenticator(&config, Arc::new(held), None))
                .with_extractor(Arc::new(BearerOrXApiKey)),
            ..Admit::over(
                &config,
                vec![("claude".to_owned(), Ok(LlmProviders::Anthropic))],
            )
        })
    }

    #[rstest]
    #[case::active(Held::Active, VIRTUAL_KEY, None)]
    #[case::revoked(Held::Revoked, VIRTUAL_KEY, Some("invalid_token"))]
    #[case::holding_what_the_native_app_does_not_serve(
        Held::Owned,
        VIRTUAL_KEY,
        Some("not_native")
    )]
    #[case::unreadable(Held::Unreachable, VIRTUAL_KEY, Some("unavailable"))]
    #[case::unknown(
        Held::Active,
        ("authorization", "Bearer sk-someone-else"),
        Some("invalid_token")
    )]
    #[tokio::test]
    async fn a_virtual_key_is_admitted_only_when_the_native_app_authenticates_it(
        #[case] held: Held,
        #[case] credential: (&str, &str),
        #[case] over: Option<&str>,
    ) {
        let refused = holding(held)
            .admit(messages(&[credential], "claude"))
            .await
            .err()
            .map(|(_, reason)| (reason.why, reason.over));
        assert_eq!(refused, over.map(|over| (Refused::Credential, over.into())));
    }

    async fn refusal(admit: Arc<Admit>, request: Request) -> Option<Refused> {
        admit
            .admit(request)
            .await
            .err()
            .map(|(_, reason)| reason.why)
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

    /// A database that answers for no end user.
    struct Unreachable;

    impl EndUserLookup for Unreachable {
        fn find<'a>(
            &'a self,
            _: &'a str,
        ) -> futures_util::future::BoxFuture<
            'a,
            Result<Option<litellm_repositories::EndUserRecord>, litellm_error::Source>,
        > {
            Box::pin(async {
                Err(litellm_error::Source::new(std::io::Error::other(
                    "database down",
                )))
            })
        }
    }

    fn unreachable() -> Option<Arc<dyn EndUserLookup>> {
        Some(Arc::new(Unreachable))
    }

    #[rstest]
    #[case::the_proxy_does_not_have(false, unreachable(), None)]
    #[case::that_cannot_be_read(true, unreachable(), Some(Refused::Body))]
    #[case::the_native_app_built_no_pool_for(true, None, Some(Refused::Body))]
    #[tokio::test]
    async fn an_end_user_is_left_to_python_by_a_database_its_row_cannot_be_read_from(
        #[case] database: bool,
        #[case] end_users: Option<Arc<dyn EndUserLookup>>,
        #[case] why: Option<Refused>,
    ) {
        let config = Config::from_yaml(CONFIG).unwrap();
        let admit = Arc::new(Admit {
            database,
            end_users,
            ..Admit::over(
                &config,
                vec![("claude".to_owned(), Ok(LlmProviders::Anthropic))],
            )
        });
        let request = Request::post("/v1/messages")
            .header(MASTER_KEY.0, MASTER_KEY.1)
            .body(Body::from(
                r#"{"model":"claude","max_tokens":8,"messages":[],"metadata":{"user_id":"customer"}}"#,
            ))
            .unwrap();

        assert_eq!(refusal(admit, request).await, why);
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

        let (returned, refused) = admit().admit(request).await.unwrap_err();

        assert_eq!(
            (
                refused.why,
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
