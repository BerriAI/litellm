//! Which requests the gateway answers itself. In front of the Python proxy every other request
//! goes to Python, and standalone answers it 501.
//!
//! [`Admit`] holds what the native app was built from, and each request is put to the three stages
//! of checks in `checks`. This is the only part that does IO: it authenticates the caller before
//! the `CALLER` stage, reads the body with `body` before the `BODY` stage, and reads the end user
//! the body names once every check has passed.

mod body;
mod checks;

use std::sync::Arc;

use axum::{body::Body, extract::Request, http::request::Parts};
use litellm_config::{Config, Environment};
use litellm_core_utils::get_llm_provider_logic::LlmProviders;
use litellm_gateway_auth::{Auth, Authenticated, Preauthenticated};
use litellm_gateway_auth_core::{Allowlists, Caller, authz::ModelList};
use litellm_gateway_inference::{JsonObject, routing};
use litellm_router_core::{Router, registry::DeploymentFilter};
use serde_json::{Map, Value};

use crate::{
    AppState, Gateway, StoredConfig,
    callbacks::parallel_request_limiter,
    host::{self, NativeHost, NativeRows, auth::AuthStore},
    middleware::request_allowlist,
};
use checks::Fields;
pub use checks::Reason;

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
    /// The router the native app serves through, which says where a request for a name can be
    /// served and what a call served there asks of its host.
    router: Router,
    /// The native host, which says which of a model group's host steps it answers.
    host: NativeHost,
    /// The proxy's address and route allowlists, read by the rule the Python proxy reads them with.
    allowlists: Allowlists,
    auth: Auth<NativeRows>,
    end_users: EndUsers,
    /// The native app's inference, which tells whether a caller may call the model it asks for.
    /// `None` leaves every caller with a model list to Python.
    inference: Option<Arc<Gateway>>,
}

/// Where the end user a body names is read from.
enum EndUsers {
    /// The store a virtual key is read from.
    Store(Arc<AuthStore>),
    /// The proxy reads no database either, so no end user has a row.
    NoDatabase,
    /// The proxy reads a database the gateway does not, so only the proxy knows what an end
    /// user's row holds.
    Unread,
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
            end_users: match (&state.auth_store, state.database_unread) {
                (Some(store), _) => EndUsers::Store(store.clone()),
                (None, false) => EndUsers::NoDatabase,
                (None, true) => EndUsers::Unread,
            },
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
            auth: host::auth::auth(config, None, false),
            end_users: EndUsers::NoDatabase,
            config: config.clone(),
            environment: Environment::default(),
            stored: StoredConfig::Unused,
            deployed,
            host: NativeHost::default(),
            inference: None,
        }
    }

    /// A caller whose model lists do not reach the model it asks for goes to Python, which
    /// answers the refusal itself. The native route asks the same question of the same router,
    /// and the auth core answers both.
    fn model_access(&self, caller: &Authenticated<NativeRows>, fields: &Fields) -> Option<Reason> {
        let model = fields.get("model")?.as_str()?;
        let allowed = match &self.inference {
            Some(inference) => routing::allows(inference, caller, model),
            None => host::auth::layers(caller).all(|layer| layer.list() == ModelList::Every),
        };
        (!allowed).then(|| Reason::new(Refused::Credential, "models"))
    }

    /// A key whose model list names a model access group goes to Python, which checks the
    /// budget the proxy's database may hold for that group (`_model_access_group_max_budget_check`).
    /// The gateway reads no group budget, so every group the router knows is taken as budgeted,
    /// even one that does not serve the model asked for.
    fn access_group_budget(&self, caller: &Authenticated<NativeRows>) -> Option<Reason> {
        let groups = self.router.registry().access_groups();
        host::auth::allowed_models(caller)
            .iter()
            .any(|name| groups.contains_key(name))
            .then(|| Reason::new(Refused::Credential, "model_access_group_budget"))
    }

    /// The caller with the end user its body names. The request goes to Python when that end
    /// user's row holds what the native app does not apply, when the row could not be read, or when
    /// the proxy reads a database the gateway cannot sign in to.
    async fn with_end_user(
        &self,
        caller: Authenticated<NativeRows>,
        fields: &Fields,
    ) -> Result<Authenticated<NativeRows>, Reason> {
        let Some(end_user_id) = checks::end_user_id(fields) else {
            return Ok(caller);
        };
        let store = match &self.end_users {
            EndUsers::Store(store) => Some(store.as_ref()),
            EndUsers::NoDatabase => None,
            EndUsers::Unread => return Err(Reason::new(Refused::Body, "end_user")),
        };
        match Box::pin(host::auth::end_user(store, end_user_id)).await {
            Ok(end_user) => Ok(Authenticated::from(Caller {
                end_user,
                ..Caller::clone(&caller)
            })),
            Err(_) => Err(Reason::new(Refused::Body, "end_user")),
        }
    }

    /// The body is read only for a request that passes every other check, and parsed once: an
    /// admitted request carries it parsed, as the [`JsonObject`] the route takes. A refused request
    /// keeps the bytes the client sent and carries nothing more. A request the proxy's allowlists
    /// leave out is admitted as sent, before its caller or body is read, as Python refuses it before
    /// it reads either, and `middleware::request_allowlist` refuses it in the native app.
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
                .or_else(|| self.access_group_budget(caller))
                .or_else(|| token_estimate(caller, parts.uri.path(), &mut fields))
        });
        if let Some(reason) = over_its_caller {
            return Err(refused(parts, body, reason));
        }
        if let Ok(caller) = authenticated {
            match self.with_end_user(caller, &fields).await {
                Ok(caller) => parts.extensions.insert(Preauthenticated(caller)),
                Err(reason) => return Err(refused(parts, body, reason)),
            };
        }
        parts.extensions.insert(JsonObject(fields));
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
                false,
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
    #[case::an_end_user("\tend-user ", Some("end-user"))]
    #[case::nobody(" ", None)]
    #[tokio::test]
    async fn an_admitted_request_carries_the_end_user_its_body_names(
        #[case] user_id: &str,
        #[case] end_user_id: Option<&str>,
    ) {
        let sent = json!({"model": "claude", "max_tokens": 8, "messages": [],
            "metadata": {"user_id": user_id}});
        let request = Request::post("/v1/messages")
            .header(MASTER_KEY.0, MASTER_KEY.1)
            .body(Body::from(sent.to_string()))
            .unwrap();

        let admitted = admit().admit(request).await.unwrap();

        let caller = admitted.extensions().get::<Preauthenticated<NativeRows>>();
        assert_eq!(
            caller.map(|caller| caller.0.end_user.end_user_id.as_deref()),
            Some(end_user_id)
        );
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
