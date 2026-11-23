//! The gateway's answers to `gateway-auth-core`'s host steps on the key path, so it authenticates
//! a virtual key with no Python behind it, the native counterpart of python-bridge's
//! `PythonAuthHost`. A key's row is read and shaped by `auth-store`, as the Python proxy's is,
//! from the proxy's database on every request.
//!
//! A step this host does not answer yet declines, which sends the request to Python. A later
//! change serves more keys by answering one more step.

#[macro_use]
mod key;

use std::{convert::Infallible, sync::Arc, time::Duration};

use litellm_auth_store::KeyDatabase;
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
use litellm_redis_store::auth_cache::CachedKey;
use litellm_repositories::Side;
use serde_json::Value;

pub use key::Key;

/// The rows the native host hands the core: a key's, and none of the objects it does not read
/// yet. `Infallible` is a row this host does not read, so no identity that needs one exists. The
/// end user's is `()`, since the key path asks for it on every request and a request that names
/// no end user reads none.
#[derive(Clone, Debug, PartialEq)]
pub struct NativeRows;

impl Rows for NativeRows {
    type Key = Key;
    type User = Infallible;
    type Team = Infallible;
    type EndUser = ();
    type ObjectPermission = Infallible;
    type Project = Infallible;
    type Membership = Infallible;
    type Org = Infallible;
    type Agent = Infallible;
}

pub struct NativeAuthHost {
    /// Where a virtual key is read from. `None` without a database, where the master key is the
    /// only caller.
    keys: Option<Arc<KeyDatabase>>,
}

/// The request needs `step`, which this host does not answer yet.
fn declined<T>(step: &'static str) -> Result<T, Error> {
    tracing::debug!(
        step,
        "the native auth host does not serve what this request needs"
    );
    Err(Error::HostDeclined)
}

fn keyed(key: &CachedKey) -> Result<Option<Keyed<Key>>, Error> {
    match key::read(key) {
        Ok(key) => Ok(Some(key)),
        Err(field) => declined(field),
    }
}

/// What the core reads of this host beside `general_settings`. A blank master key is none.
/// `database` is whether this host reads keys from one, and without one the core refuses every
/// token but the master key before it asks for a key. The gateway guards its inference routes
/// alone, so it names no route group, and it serves no UI login. The proxy budget, the license
/// and the environment only shape steps and text this host does not answer with.
fn settings(config: &Config, database: bool) -> Settings {
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
            has_db: database,
            db_configured: database,
            ui_login: false,
            proxy_budget: false,
            metrics_require_auth: true,
            route_groups: RouteGroups::default(),
        },
        &|_| None,
    )
}

/// Who the inference routes and `admit` authenticate: the callers the core finds over this host
/// for `config`, the master key and with `keys` a virtual key read from them, taken from the
/// header an inference client sends its key in.
pub fn auth(config: &Config, keys: Option<Arc<KeyDatabase>>) -> Auth<NativeRows> {
    let settings = settings(config, keys.is_some());
    let keys = KeyAuthenticator::new(NativeAuthHost { keys }, settings, Arc::new(SystemClock));
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

    /// The gateway shares no cache, so every key is read from the database.
    async fn cached_key(&self, _: &KeyHash) -> Result<Option<Keyed<Key>>, Error> {
        Ok(None)
    }

    async fn db_key(&self, hash: &KeyHash, _: &()) -> Result<Option<Keyed<Key>>, Error> {
        let Some(keys) = &self.keys else {
            return declined("db_key");
        };
        match Box::pin(keys.read(hash, Side::Read)).await {
            Ok(Some(key)) => keyed(&key),
            Ok(None) => Ok(None),
            Err(error) => {
                tracing::warn!(
                    error = %litellm_error::Report::new(&error),
                    "a virtual key was not read from the database"
                );
                Err(Error::HostFailed)
            }
        }
    }

    async fn key_by_hash(&self, _: &KeyHash) -> Result<Option<MappedKey<Key>>, Error> {
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

    async fn ui_key(&self, _: &Token) -> Result<Option<Keyed<Key>>, Error> {
        declined("ui_key")
    }

    /// The gateway keeps no key in memory, so it holds none to evict.
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

    /// The gateway reads each object when the core asks for it, with nothing to warm.
    async fn prefetch_identity(&self, _: Option<&Token>, _: Option<&str>) -> Result<(), Error> {
        Ok(())
    }

    async fn prefetch_auth_objects(&self, _: &Key, _: &()) -> Result<(), Error> {
        Ok(())
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

    async fn normalize_model(&self, key: &Key) -> Result<(), Error> {
        match key.aliased {
            true => declined("aliases"),
            false => Ok(()),
        }
    }

    /// A temporary increase is written in the key's metadata, which a key this host read does not
    /// hold.
    async fn with_temp_budget(&self, key: Keyed<Key>) -> Result<Keyed<Key>, Error> {
        Ok(key)
    }

    /// No request is free here, so [`Self::key_budgets`] is asked for every key.
    async fn key_skips_budget_checks(&self, _: &Key) -> Result<bool, Error> {
        Ok(false)
    }

    async fn team_model_access(&self, team: &Infallible, _: &str) -> Result<ModelAccess, Error> {
        match *team {}
    }

    /// A key this host read has no team and no owner, so no membership budgets it.
    async fn team_member_budget(&self, _: &Key, _: Option<&Infallible>) -> Result<(), Error> {
        Ok(())
    }

    async fn key_budgets(&self, key: &Key, _: Option<&Infallible>, _: bool) -> Result<(), Error> {
        match key.budget {
            Some(column) => declined(column),
            None => Ok(()),
        }
    }

    async fn proxy_budget_alert(&self, _: &Key) -> Result<(), Error> {
        Ok(())
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

    async fn finish_jwt(&self, _: FinishedJwt<'_, NativeRows>) -> Result<Key, Error> {
        declined("finish_jwt")
    }

    async fn with_team(&self, _: &Key, team: &Infallible) -> Result<Key, Error> {
        match *team {}
    }
}

#[cfg(test)]
#[path = "../../../tests/keys/mod.rs"]
pub(crate) mod keys;

#[cfg(test)]
mod tests {
    use axum::{
        Router,
        body::Body,
        http::{Request, StatusCode, request::Parts},
        middleware::from_extractor_with_state,
        routing::get,
    };
    use jiff::Timestamp;
    use litellm_gateway_auth::{Authenticated, Error as Rejected, RequireMasterKey};
    use litellm_gateway_auth_core::Caller;
    use rstest::rstest;
    use serde_json::{Value as Columns, json};
    use tower::ServiceExt;

    use super::{keys::Held, *};

    const MASTER_KEY: &str = "sk-master-credential";
    const VIRTUAL_KEY: &str = "sk-test-credential";
    const FAR_FUTURE: &str = "2100-01-01T00:00:00";

    /// The rows a key's column may point at.
    const PARENTS: &str = r#"
        INSERT INTO "LiteLLM_BudgetTable" (budget_id, created_by, updated_by) VALUES ('budget', 'test', 'test');
        INSERT INTO "LiteLLM_OrganizationTable" (organization_id, organization_alias, budget_id, created_by, updated_by)
            VALUES ('org', 'Org', 'budget', 'test', 'test');
        INSERT INTO "LiteLLM_TeamTable" (team_id) VALUES ('team');
        INSERT INTO "LiteLLM_ProjectTable" (project_id, project_alias, team_id, models, created_by, updated_by)
            VALUES ('project', 'Project', 'team', ARRAY[]::text[], 'test', 'test');
        INSERT INTO "LiteLLM_ObjectPermissionTable" (object_permission_id) VALUES ('permission');
    "#;

    fn config(general_settings: &str) -> Config {
        Config::from_yaml(&format!(
            "model_list: []\ngeneral_settings:\n{general_settings}"
        ))
        .unwrap()
    }

    fn with_master_key() -> Config {
        config(&format!("  master_key: {MASTER_KEY}\n"))
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
        keys: Option<KeyDatabase>,
        header: (&str, &str),
    ) -> Result<Authenticated<NativeRows>, Rejected> {
        auth(config, keys.map(Arc::new))
            .authenticate_parts(&parts(header))
            .await
    }

    /// What `token` gets as a bearer from a gateway that reads its keys from `keys`.
    async fn presenting(
        keys: KeyDatabase,
        token: &str,
    ) -> Result<Authenticated<NativeRows>, Rejected> {
        let bearer = format!("Bearer {token}");
        authenticate(&with_master_key(), Some(keys), ("authorization", &bearer)).await
    }

    /// The database holding [`VIRTUAL_KEY`] alone, a plain key with `columns` set on it.
    async fn holding(columns: Columns) -> Held {
        let held = Held::start().await;
        held.execute(PARENTS).await;
        held.key(VIRTUAL_KEY, columns).await;
        held
    }

    /// What [`VIRTUAL_KEY`] gets from a gateway that shares no cache, with `columns` set on it.
    async fn authenticate_holding(columns: Columns) -> Result<Authenticated<NativeRows>, Rejected> {
        let held = holding(columns).await;
        presenting(held.database(), VIRTUAL_KEY).await
    }

    #[rstest]
    #[case::without_a_database(None)]
    #[case::with_a_database_that_is_never_read(Some(keys::unreachable()))]
    #[tokio::test]
    async fn the_master_key_is_the_admin_caller_with_and_without_a_database(
        #[case] keys: Option<KeyDatabase>,
        #[values(("authorization", "Bearer sk-master-credential"), ("x-api-key", MASTER_KEY))]
        header: (&str, &str),
    ) {
        let caller = authenticate(&with_master_key(), keys, header)
            .await
            .unwrap();

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
    async fn without_a_database_every_token_but_the_master_key_is_refused_as_invalid(
        #[case] header: (&str, &str),
        #[case] refused: &str,
    ) {
        let rejected = authenticate(&with_master_key(), None, header)
            .await
            .err()
            .unwrap();

        assert_eq!(<&'static str>::from(rejected), refused);
    }

    #[rstest]
    #[case::unset("  master_key: null\n")]
    #[case::empty("  master_key: ''\n")]
    #[case::blank("  master_key: '   '\n")]
    #[tokio::test]
    async fn no_token_authenticates_while_no_master_key_is_configured(#[case] master_key: &str) {
        let rejected = authenticate(
            &config(master_key),
            None,
            ("authorization", "Bearer sk-anything"),
        )
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
                &config, None,
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

    #[rstest]
    #[case::as_generated(json!({}), None)]
    #[case::no_key_type(json!({"key_type": null}), None)]
    #[case::unblocked(json!({"blocked": false}), None)]
    #[case::expiring(json!({"expires": FAR_FUTURE}), Some("2100-01-01T00:00:00Z"))]
    #[case::empty_json_columns(json!({"budget_limits": null, "aliases": {}, "metadata": {}}), None)]
    #[tokio::test(flavor = "multi_thread")]
    async fn a_key_that_holds_only_what_the_native_app_honors_is_a_caller_named_by_its_stored_hash(
        #[case] columns: Columns,
        #[case] expires: Option<&str>,
    ) {
        let caller = authenticate_holding(columns).await.unwrap();

        let Caller::Key(key) = &*caller else {
            panic!("the key authenticated as the master key");
        };
        assert_eq!(
            (caller.hash(), caller.role(), key.key.record.expires),
            (
                Some(&Token::new(VIRTUAL_KEY).hash()),
                LitellmRole::InternalUser,
                expires.map(|at| at.parse::<Timestamp>().unwrap()),
            )
        );
    }

    #[rstest]
    #[case::key_type(json!({"key_type": "management"}))]
    #[case::soft_budget_cooldown(json!({"soft_budget_cooldown": true}))]
    #[case::models(json!({"models": ["claude"]}))]
    #[case::aliases(json!({"aliases": {"a": "b"}}))]
    #[case::config(json!({"config": {"a": 1}}))]
    #[case::router_settings(json!({"router_settings": {"a": 1}}))]
    #[case::user_id(json!({"user_id": "user"}))]
    #[case::an_empty_user_id(json!({"user_id": ""}))]
    #[case::team_id(json!({"team_id": "team"}))]
    #[case::the_ui_team(json!({"team_id": "litellm-dashboard"}))]
    #[case::agent_id(json!({"agent_id": "agent"}))]
    #[case::project_id(json!({"project_id": "project"}))]
    #[case::org_id(json!({"organization_id": "org"}))]
    #[case::permissions(json!({"permissions": {"a": true}}))]
    #[case::max_parallel_requests(json!({"max_parallel_requests": 1}))]
    #[case::metadata(json!({"metadata": {"guardrails": ["pii"]}}))]
    #[case::tpm_limit(json!({"tpm_limit": 1}))]
    #[case::rpm_limit(json!({"rpm_limit": 1}))]
    #[case::a_limit_of_zero(json!({"rpm_limit": 0}))]
    #[case::tpd_limit(json!({"tpd_limit": 1}))]
    #[case::max_budget(json!({"max_budget": 5.0}))]
    #[case::budget_duration(json!({"budget_duration": "30d"}))]
    #[case::budget_reset_at(json!({"budget_reset_at": "1970-01-01T00:00:09"}))]
    #[case::budget_limits(json!({"budget_limits": [{"a": 1}]}))]
    #[case::budget_fallbacks(json!({"budget_fallbacks": {"a": ["b"]}}))]
    #[case::budget_id(json!({"budget_id": "budget"}))]
    #[case::model_max_budget(json!({"model_max_budget": {"claude": 1}}))]
    #[case::allowed_cache_controls(json!({"allowed_cache_controls": ["no-cache"]}))]
    #[case::allowed_routes(json!({"allowed_routes": ["/v1/messages"]}))]
    #[case::access_group_ids(json!({"access_group_ids": ["group"]}))]
    #[case::object_permission_id(json!({"object_permission_id": "permission"}))]
    #[tokio::test(flavor = "multi_thread")]
    async fn a_key_that_holds_anything_else_is_not_native(#[case] columns: Columns) {
        let refused = authenticate_holding(columns).await;

        assert!(
            matches!(refused, Err(Rejected::NotNative)),
            "{:?}",
            refused.err()
        );
    }

    #[rstest]
    #[case::alone(json!({"blocked": true}))]
    #[case::of_a_team(json!({"blocked": true, "team_id": "team"}))]
    #[tokio::test(flavor = "multi_thread")]
    async fn a_blocked_key_is_an_invalid_token(#[case] columns: Columns) {
        let refused = authenticate_holding(columns).await;

        assert!(
            matches!(refused, Err(Rejected::InvalidToken)),
            "{:?}",
            refused.err()
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn a_key_the_database_does_not_hold_is_an_invalid_token() {
        let held = Held::holding("sk-another-credential", json!({})).await;

        let refused = presenting(held.database(), VIRTUAL_KEY).await;

        assert!(
            matches!(refused, Err(Rejected::InvalidToken)),
            "{:?}",
            refused.err()
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn an_expired_key_is_expired() {
        let refused = authenticate_holding(json!({"expires": "1970-01-01T00:00:01"})).await;

        assert!(
            matches!(refused, Err(Rejected::Expired)),
            "{:?}",
            refused.err()
        );
    }

    #[rstest]
    #[case::the_master_key(MASTER_KEY, true)]
    #[case::a_token_that_is_not_a_key("eyJhbGciOiJSUzI1NiJ9.e30.c2ln", false)]
    #[tokio::test(flavor = "multi_thread")]
    async fn only_a_virtual_key_is_read_from_the_database(
        #[case] token: &str,
        #[case] authenticated: bool,
    ) {
        let held = holding(json!({})).await;

        let caller = presenting(held.database(), token).await;

        assert_eq!(
            caller.ok().map(|caller| *caller == Caller::MasterKey),
            authenticated.then_some(true)
        );
        assert_eq!(held.key_reads(), 0);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn a_database_that_fails_is_unavailable_and_never_a_denial() {
        let Err(refused) = presenting(keys::unreachable(), VIRTUAL_KEY).await else {
            panic!("a key was authenticated with no database to read it from");
        };

        assert!(matches!(refused, Rejected::Unavailable(_)), "{refused:?}");
        assert!(!refused.is_denied());
    }
}
