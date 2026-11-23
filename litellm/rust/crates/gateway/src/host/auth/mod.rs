//! The gateway's answers to `gateway-auth-core`'s host steps on the key path, so the core decides
//! who calls it with no Python behind it, the native counterpart of python-bridge's
//! `PythonAuthHost`.
//!
//! This host reads no key yet. It answers the steps the master key takes and declines every
//! other one, so the master key is the only caller the core hands back. Serving a virtual key
//! starts by giving [`NativeRows::Key`] a row.

use std::{convert::Infallible, sync::Arc, time::Duration};

use litellm_config::{Config, LitellmRole};
use litellm_gateway_auth::{Auth, BearerOrXApiKey, KeyAuthenticator, SystemClock};
use litellm_gateway_auth_core::{
    AgentIdentityFailure, AuthHost, Error, HostFacts, KeyHash, MasterKey, OperatorHooks,
    RouteGroups, Rows, Settings, Token,
    authn::{
        jwt::{KeyMappingRow, KeyMappingWrite, MappedClaim, MappedKey},
        key::{CachedTeam, Keyed},
    },
    identity::{
        jwt::{
            AliasRead, DelegatedTeam, FinishedJwt, IdentityRead, ManagedAgent, ModelAccess, Org,
            RegisteredAgent, TeamRead, UserLookup, UserRead,
        },
        key::Owner,
    },
};
use serde_json::Value;

/// The rows the native host hands the core. `Infallible` is a row this host does not read, so no
/// identity that needs one exists. The end user's is `()`, since the key path asks for it on
/// every request and a request that names no end user reads none.
#[derive(Clone, Debug, PartialEq)]
pub struct NativeRows;

impl Rows for NativeRows {
    type Key = Infallible;
    type User = Infallible;
    type Team = Infallible;
    type EndUser = ();
    type ObjectPermission = Infallible;
    type Project = Infallible;
    type Membership = Infallible;
    type Org = Infallible;
    type Agent = Infallible;
}

pub struct NativeAuthHost;

/// The request needs `step`, which this host does not answer.
fn declined<T>(step: &'static str) -> Result<T, Error> {
    tracing::debug!(
        step,
        "the native auth host does not serve what this request needs"
    );
    Err(Error::HostFailed)
}

/// What the core reads of this host beside `general_settings`. A blank master key is none. The
/// host reads no key, which is what having no database says to the core, so it refuses every
/// token but the master key before it asks for one. The gateway guards its inference routes
/// alone, so it names no route group, and it serves no UI login.
fn settings(config: &Config) -> Settings {
    let master_key = config
        .general_settings
        .master_key
        .as_ref()
        .filter(|key| !key.expose().trim().is_empty());
    Settings::new(
        &config.general_settings,
        HostFacts {
            master_key: master_key.map(|key| MasterKey::new(key.clone())),
            premium: false,
            has_db: false,
            db_configured: false,
            ui_login: false,
            proxy_budget: false,
            metrics_require_auth: true,
            route_groups: RouteGroups::default(),
        },
        &|_| None,
    )
}

/// Who the inference routes and `admit` authenticate: the callers the core finds over this host
/// for `config`, read from the header an inference client sends its key in.
pub fn auth(config: &Config) -> Auth<NativeRows> {
    let keys = KeyAuthenticator::new(NativeAuthHost, settings(config), Arc::new(SystemClock));
    Auth::new(Arc::new(keys)).with_extractor(Arc::new(BearerOrXApiKey))
}

struct NoOperatorCode;

impl OperatorHooks for NoOperatorCode {
    async fn custom_validate(&self, _: &Value) -> Result<bool, Error> {
        declined("custom_validate")
    }
}

impl AuthHost for NativeAuthHost {
    type Rows = NativeRows;

    /// The gateway holds no key, so none is cached under any hash, the master key's included.
    async fn cached_key(&self, _: &KeyHash) -> Result<Option<Keyed<Infallible>>, Error> {
        Ok(None)
    }

    async fn db_key(&self, _: &KeyHash, _: &()) -> Result<Option<Keyed<Infallible>>, Error> {
        declined("db_key")
    }

    async fn key_by_hash(&self, _: &KeyHash) -> Result<Option<MappedKey<Infallible>>, Error> {
        declined("key_by_hash")
    }

    async fn key_mapping_hash(
        &self,
        _: &str,
        _: &str,
        _: Option<&str>,
    ) -> Result<Option<KeyHash>, Error> {
        declined("key_mapping_hash")
    }

    async fn reusable_key_hash(&self, _: &str, _: Option<&str>) -> Result<Option<KeyHash>, Error> {
        declined("reusable_key_hash")
    }

    async fn ui_key(&self, _: &Token) -> Result<Option<Keyed<Infallible>>, Error> {
        declined("ui_key")
    }

    async fn evict_key(&self, _: &KeyHash) -> Result<(), Error> {
        Ok(())
    }

    async fn fetch_json(&self, _: &str) -> Result<Option<Value>, Error> {
        declined("fetch_json")
    }

    async fn fetch_userinfo(&self, _: &str, _: &Token) -> Result<Option<Value>, Error> {
        declined("fetch_userinfo")
    }

    async fn cached_document(&self, _: &str) -> Result<Option<Value>, Error> {
        declined("cached_document")
    }

    async fn cache_document(&self, _: &str, _: &Value, _: Duration) -> Result<(), Error> {
        declined("cache_document")
    }

    async fn cached_key_mapping(&self, _: MappedClaim<'_>) -> Result<Option<Value>, Error> {
        declined("cached_key_mapping")
    }

    async fn cache_key_mapping(
        &self,
        _: MappedClaim<'_>,
        _: &Value,
        _: Duration,
    ) -> Result<(), Error> {
        declined("cache_key_mapping")
    }

    async fn forget_key_mapping(&self, _: MappedClaim<'_>) -> Result<(), Error> {
        declined("forget_key_mapping")
    }

    fn operator_hooks(&self) -> Option<&impl OperatorHooks> {
        None::<&NoOperatorCode>
    }

    async fn managed_agent(&self, _: &Value) -> Result<Option<ManagedAgent<Infallible>>, Error> {
        declined("managed_agent")
    }

    async fn registered_agent(&self, _: &str) -> Result<Option<RegisteredAgent>, Error> {
        declined("registered_agent")
    }

    async fn delegated_agent_team(&self, _: &DelegatedTeam<'_>) -> Result<Option<String>, Error> {
        declined("delegated_agent_team")
    }

    async fn key_owner(&self, _: &str) -> Result<Option<Owner<Infallible>>, Error> {
        declined("key_owner")
    }

    async fn user(&self, _: &UserLookup<'_>) -> Result<UserRead<Infallible>, Error> {
        declined("user")
    }

    async fn resolve_user_identity(
        &self,
        _: &UserLookup<'_>,
        _: Option<&str>,
    ) -> Result<IdentityRead<NativeRows>, Error> {
        declined("resolve_user_identity")
    }

    async fn cached_team(&self, _: &str) -> Result<Option<CachedTeam<Infallible>>, Error> {
        declined("cached_team")
    }

    async fn key_team(&self, _: &str) -> Result<Option<Infallible>, Error> {
        declined("key_team")
    }

    async fn team(&self, _: &str, _: bool) -> Result<TeamRead<Infallible>, Error> {
        declined("team")
    }

    async fn granting_team(&self, _: &str) -> Result<TeamRead<Infallible>, Error> {
        declined("granting_team")
    }

    async fn team_by_alias(&self, _: &str) -> Result<AliasRead<Infallible>, Error> {
        declined("team_by_alias")
    }

    async fn load_team_membership(&self, _: &str, _: &str) -> Result<Option<Infallible>, Error> {
        declined("load_team_membership")
    }

    async fn org(&self, _: &str) -> Result<Option<Org<Infallible>>, Error> {
        declined("org")
    }

    async fn org_by_alias(&self, _: &str) -> Result<Option<Org<Infallible>>, Error> {
        declined("org_by_alias")
    }

    async fn jwt_end_user(&self, _: &str) -> Result<(), Error> {
        declined("jwt_end_user")
    }

    /// A request that names no end user reads none.
    async fn end_user(&self, end_user_id: Option<&str>) -> Result<(), Error> {
        match end_user_id {
            None => Ok(()),
            Some(_) => declined("end_user"),
        }
    }

    async fn validated_end_user_id(&self, _: &str) -> Result<Option<String>, Error> {
        declined("validated_end_user_id")
    }

    async fn object_permission(&self, _: &str) -> Result<Option<Infallible>, Error> {
        declined("object_permission")
    }

    async fn project(&self, _: &str) -> Result<Option<Infallible>, Error> {
        declined("project")
    }

    async fn prefetch_identity(&self, _: Option<&Token>, _: Option<&str>) -> Result<(), Error> {
        Ok(())
    }

    async fn prefetch_auth_objects(&self, key: &Infallible, _: &()) -> Result<(), Error> {
        match *key {}
    }

    async fn set_user_role(&self, user: &Infallible, _: LitellmRole) -> Result<Infallible, Error> {
        match *user {}
    }

    async fn patch_team_membership(
        &self,
        user: &Infallible,
        _: &[String],
        _: &[String],
    ) -> Result<Infallible, Error> {
        match *user {}
    }

    async fn normalize_model(&self, key: &Infallible) -> Result<(), Error> {
        match *key {}
    }

    async fn with_temp_budget(&self, key: Keyed<Infallible>) -> Result<Keyed<Infallible>, Error> {
        match key.row {}
    }

    async fn key_skips_budget_checks(&self, key: &Infallible) -> Result<bool, Error> {
        match *key {}
    }

    async fn team_model_access(&self, team: &Infallible, _: &str) -> Result<ModelAccess, Error> {
        match *team {}
    }

    async fn team_member_budget(
        &self,
        key: &Infallible,
        _: Option<&Infallible>,
    ) -> Result<(), Error> {
        match *key {}
    }

    async fn key_budgets(
        &self,
        key: &Infallible,
        _: Option<&Infallible>,
        _: bool,
    ) -> Result<(), Error> {
        match *key {}
    }

    async fn proxy_budget_alert(&self, key: &Infallible) -> Result<(), Error> {
        match *key {}
    }

    async fn record_agent_authentication(
        &self,
        agent: &Infallible,
    ) -> Result<Option<AgentIdentityFailure>, Error> {
        match *agent {}
    }

    async fn add_team_member(&self, _: &str, _: &str) -> Result<(), Error> {
        declined("add_team_member")
    }

    async fn create_key_mapping(&self, _: &KeyMappingRow<'_>) -> Result<KeyMappingWrite, Error> {
        declined("create_key_mapping")
    }

    async fn finish_jwt(&self, _: FinishedJwt<'_, NativeRows>) -> Result<Infallible, Error> {
        declined("finish_jwt")
    }

    async fn with_team(&self, key: &Infallible, _: &Infallible) -> Result<Infallible, Error> {
        match *key {}
    }
}

#[cfg(test)]
mod tests {
    use axum::{
        Router,
        body::Body,
        http::{Request, StatusCode, request::Parts},
        middleware::from_extractor_with_state,
        routing::get,
    };
    use litellm_gateway_auth::{Authenticated, Error as Rejected, RequireMasterKey};
    use litellm_gateway_auth_core::Caller;
    use rstest::rstest;
    use tower::ServiceExt;

    use super::*;

    const MASTER_KEY: &str = "sk-master-credential";

    fn config(general_settings: &str) -> Config {
        Config::from_yaml(&format!(
            "model_list: []\ngeneral_settings:\n{general_settings}"
        ))
        .unwrap()
    }

    fn parts(header: (&str, &str)) -> Parts {
        Request::post("/v1/messages")
            .header(header.0, header.1)
            .body(())
            .unwrap()
            .into_parts()
            .0
    }

    async fn authenticate(
        config: &Config,
        header: (&str, &str),
    ) -> Result<Authenticated<NativeRows>, Rejected> {
        auth(config).authenticate_parts(&parts(header)).await
    }

    #[rstest]
    #[case::without_a_database("")]
    #[case::with_a_database("  database_url: postgresql://user:password@localhost:5432/litellm\n")]
    #[tokio::test]
    async fn the_master_key_is_the_admin_caller_with_and_without_a_database(
        #[case] database: &str,
        #[values(("authorization", "Bearer sk-master-credential"), ("x-api-key", MASTER_KEY))]
        header: (&str, &str),
    ) {
        let config = config(&format!("  master_key: {MASTER_KEY}\n{database}"));

        let caller = authenticate(&config, header).await.unwrap();

        assert_eq!(
            (&*caller, caller.role()),
            (&Caller::MasterKey, LitellmRole::ProxyAdmin)
        );
    }

    #[rstest]
    #[case::another_virtual_key(("authorization", "Bearer sk-someone-else"), "invalid_token")]
    #[case::a_token_that_is_no_key(("authorization", "Bearer someone-else"), "invalid_token")]
    #[case::a_jwt(("authorization", "Bearer a.b.c"), "invalid_token")]
    #[case::another_key_in_the_key_header(("x-api-key", "sk-someone-else"), "invalid_api_key")]
    #[tokio::test]
    async fn every_token_but_the_master_key_is_refused_as_invalid(
        #[case] header: (&str, &str),
        #[case] refused: &str,
        #[values(
            "",
            "  database_url: postgresql://user:password@localhost:5432/litellm\n"
        )]
        database: &str,
    ) {
        let config = config(&format!("  master_key: {MASTER_KEY}\n{database}"));

        let rejected = authenticate(&config, header).await.err().unwrap();

        assert_eq!(<&'static str>::from(rejected), refused);
    }

    #[rstest]
    #[case::unset("  master_key: null\n")]
    #[case::empty("  master_key: ''\n")]
    #[case::blank("  master_key: '   '\n")]
    #[tokio::test]
    async fn no_token_authenticates_while_no_master_key_is_configured(#[case] master_key: &str) {
        let rejected = authenticate(&config(master_key), ("authorization", "Bearer sk-anything"))
            .await
            .err()
            .unwrap();

        assert_eq!(<&'static str>::from(rejected), "unconfigured");
    }

    #[rstest]
    #[case::the_master_key(MASTER_KEY, Some("Bearer sk-master-credential"), 204)]
    #[case::another_token(MASTER_KEY, Some("Bearer other-key"), 401)]
    #[case::no_header(MASTER_KEY, None, 401)]
    #[case::another_scheme(MASTER_KEY, Some("Basic sk-master-credential"), 401)]
    #[case::an_empty_token(MASTER_KEY, Some("Bearer "), 401)]
    #[case::an_empty_master_key("", Some("Bearer "), 500)]
    #[case::a_blank_master_key("   ", Some("Bearer "), 500)]
    #[tokio::test]
    async fn a_route_for_the_master_key_alone_never_shows_the_key(
        #[case] master_key: &str,
        #[case] authorization: Option<&str>,
        #[case] status: u16,
    ) {
        let config = config(&format!("  master_key: '{master_key}'\n"));
        let app = Router::new()
            .route("/protected", get(|| async { StatusCode::NO_CONTENT }))
            .layer(from_extractor_with_state::<RequireMasterKey, _>(auth(
                &config,
            )));
        let request = Request::get("/protected");
        let request = match authorization {
            Some(value) => request.header("authorization", value),
            None => request,
        };

        let response = app
            .oneshot(request.body(Body::empty()).unwrap())
            .await
            .unwrap();

        assert_eq!(response.status().as_u16(), status);
        let body = axum::body::to_bytes(response.into_body(), 4096)
            .await
            .unwrap();
        assert!(!std::str::from_utf8(&body).unwrap().contains(MASTER_KEY));
    }
}
