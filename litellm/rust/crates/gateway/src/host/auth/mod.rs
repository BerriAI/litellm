//! The gateway's answers to `gateway-auth-core`'s host steps on the key path, so it authenticates
//! a virtual key with no Python behind it, the native counterpart of python-bridge's
//! `PythonAuthHost`. A key's row is read and shaped by `auth-store`, as the Python proxy's is,
//! and `store.rs` puts the auth cache in front of it, as `UserApiKeyCache` sits in front of
//! Python's: this process's memory, and the Redis shared with the Python pods when
//! `enable_redis_auth_cache` attaches it, then the proxy's database on a miss, which fills it.
//!
//! A step this host does not answer yet declines, which sends the request to Python. A later
//! change serves more keys by answering one more step.

#[macro_use]
mod key;
mod store;

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
        key::{Owner, ProjectRecord},
    },
};
use litellm_redis_store::auth_cache::CachedKey;
use serde_json::Value;

pub use key::Key;
pub use store::AuthStore;
pub(crate) use store::KEY_TTL;

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
    type Membership = Infallible;
}

pub struct NativeAuthHost {
    /// Where a virtual key is read from. `None` without a database, where the master key is the
    /// only caller.
    keys: Option<Arc<AuthStore>>,
    /// The hash Python caches the master key's identity under, which is no key row.
    master: Option<KeyHash>,
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

/// The master key of `config`, which a blank one is not.
fn master_key(config: &Config) -> Option<&litellm_secret_value::SecretValue> {
    config
        .general_settings
        .master_key
        .as_ref()
        .filter(|key| !key.expose().trim().is_empty())
}

/// What the core reads of this host beside `general_settings`. `database` is whether this host reads keys from one, and without one the core refuses every
/// token but the master key before it asks for a key. The gateway guards its inference routes
/// alone, so it names no route group, and it serves no UI login. The proxy budget, the license
/// and the environment only shape steps and text this host does not answer with.
fn settings(config: &Config, database: bool) -> Settings {
    Settings::new(
        &config.general_settings,
        HostFacts {
            master_key: master_key(config).map(|key| MasterKey::new(key.clone())),
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
pub fn auth(config: &Config, keys: Option<Arc<AuthStore>>) -> Auth<NativeRows> {
    let settings = settings(config, keys.is_some());
    let host = NativeAuthHost {
        keys,
        master: master_key(config).map(|key| Token::new(key.clone()).hash()),
    };
    let keys = KeyAuthenticator::new(host, settings, Arc::new(SystemClock));
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

    async fn cached_key(&self, hash: &KeyHash) -> Result<Option<Keyed<Key>>, Error> {
        if self.master.as_ref() == Some(hash) {
            return Ok(None);
        }
        let Some(store) = &self.keys else {
            return Ok(None);
        };
        match Box::pin(store.cached(hash)).await {
            Some(key) => keyed(&key),
            None => Ok(None),
        }
    }

    /// The core asks here only after `cached_key` missed, so the cache is not read again. The
    /// store caches the key it found, whoever answers the request.
    async fn db_key(&self, hash: &KeyHash, _: &()) -> Result<Option<Keyed<Key>>, Error> {
        let Some(store) = &self.keys else {
            return declined("db_key");
        };
        match Box::pin(store.loaded(hash)).await {
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

    async fn managed_agent(&self, _: &Value) -> Result<Option<ManagedAgent>, Error> {
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

    async fn org(&self, _: &str) -> Result<Option<Org>, Error> {
        declined("org")
    }

    async fn org_by_alias(&self, _: &str) -> Result<Option<Org>, Error> {
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

    async fn project(&self, _: &str) -> Result<Option<ProjectRecord>, Error> {
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
        _: &ManagedAgent,
    ) -> Result<Option<AgentIdentityFailure>, Error> {
        declined("record_agent_authentication")
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
    use std::collections::BTreeMap;

    use axum::{
        Router,
        body::Body,
        http::{Request, StatusCode, request::Parts},
        middleware::from_extractor_with_state,
        routing::get,
    };
    use jiff::Timestamp;
    use litellm_auth_store::KeyDatabase;
    use litellm_gateway_auth::{Authenticated, Error as Rejected, RequireMasterKey};
    use litellm_gateway_auth_core::Caller;
    use litellm_redis_store::cache_invalidation::CacheInvalidation;
    use litellm_redis_testing::{RedisServer, eventually};
    use rstest::rstest;
    use serde_json::{Value as Columns, json};
    use tower::ServiceExt;

    use super::{keys::Held, *};
    use crate::callbacks::parallel_request_limiter::{KeyLimits, ModelLimits};

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

    /// The store a gateway with no coordination Redis reads `database` through.
    fn memory_only(database: KeyDatabase) -> Arc<AuthStore> {
        Arc::new(AuthStore::new(keys::memory_cache(), database))
    }

    async fn authenticate(
        config: &Config,
        keys: Option<KeyDatabase>,
        header: (&str, &str),
    ) -> Result<Authenticated<NativeRows>, Rejected> {
        auth(config, keys.map(memory_only))
            .authenticate_parts(&parts(header))
            .await
    }

    /// What `token` gets as a bearer from `auth`.
    async fn bearing(
        auth: &Auth<NativeRows>,
        token: &str,
    ) -> Result<Authenticated<NativeRows>, Rejected> {
        let bearer = format!("Bearer {token}");
        auth.authenticate_parts(&parts(("authorization", &bearer)))
            .await
    }

    /// What `token` gets as a bearer from a gateway that reads its keys from `database`.
    async fn presenting(
        database: KeyDatabase,
        token: &str,
    ) -> Result<Authenticated<NativeRows>, Rejected> {
        bearing(
            &auth(&with_master_key(), Some(memory_only(database))),
            token,
        )
        .await
    }

    /// The database holding [`VIRTUAL_KEY`] alone, a plain key with `columns` set on it.
    async fn holding(columns: Columns) -> Held {
        let held = Held::start().await;
        held.execute(PARENTS).await;
        held.key(VIRTUAL_KEY, columns).await;
        held
    }

    /// What [`VIRTUAL_KEY`] gets from a gateway with no coordination Redis, with `columns` set on
    /// it.
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
    #[case::as_generated(json!({}), None, &[])]
    #[case::no_key_type(json!({"key_type": null}), None, &[])]
    #[case::unblocked(json!({"blocked": false}), None, &[])]
    #[case::expiring(json!({"expires": FAR_FUTURE}), Some("2100-01-01T00:00:00Z"), &[])]
    #[case::with_a_model_list(json!({"models": ["claude", "haiku"]}), None, &["claude", "haiku"])]
    #[case::reaching_every_model(
        json!({"models": ["*", "all-proxy-models"]}),
        None,
        &["*", "all-proxy-models"]
    )]
    #[case::empty_json_columns(json!({"budget_limits": null, "aliases": {}, "metadata": {}}), None, &[])]
    #[tokio::test(flavor = "multi_thread")]
    async fn a_key_that_holds_only_what_the_native_app_honors_is_a_caller_named_by_its_stored_hash(
        #[case] columns: Columns,
        #[case] expires: Option<&str>,
        #[case] models: &[&str],
    ) {
        let caller = authenticate_holding(columns).await.unwrap();

        let Caller::Key(key) = &*caller else {
            panic!("the key authenticated as the master key");
        };
        assert_eq!(
            (
                caller.hash(),
                caller.role(),
                key.key.record.expires,
                key.key.models().to_vec(),
            ),
            (
                Some(&Token::new(VIRTUAL_KEY).hash()),
                LitellmRole::InternalUser,
                expires.map(|at| at.parse::<Timestamp>().unwrap()),
                models.iter().map(|model| (*model).to_owned()).collect(),
            )
        );
    }

    #[rstest]
    #[case::key_type(json!({"key_type": "management"}))]
    #[case::soft_budget_cooldown(json!({"soft_budget_cooldown": true}))]
    #[case::a_model_pattern(json!({"models": ["claude", "anthropic/*"]}))]
    #[case::the_team_models(json!({"models": ["all-team-models"]}))]
    #[case::no_default_models(json!({"models": ["no-default-models"]}))]
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
    #[case::metadata(json!({"metadata": {"guardrails": ["pii"]}}))]
    #[case::metadata_beside_a_model_limit(json!({"metadata": {"model_rpm_limit": {"claude": 2}, "tags": ["a"]}}))]
    #[case::a_model_limit_that_is_no_whole_number(json!({"metadata": {"model_tpm_limit": {"claude": "100"}}}))]
    #[case::model_limits_that_are_no_map(json!({"metadata": {"model_rpm_limit": 5}}))]
    #[case::a_priority_that_is_no_text(json!({"metadata": {"priority": 5}}))]
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
    #[case::rpm(json!({"rpm_limit": 2}), KeyLimits { rpm: Some(2), ..KeyLimits::default() })]
    #[case::tpm(json!({"tpm_limit": 100}), KeyLimits { tpm: Some(100), ..KeyLimits::default() })]
    #[case::parallel(
        json!({"max_parallel_requests": 1}),
        KeyLimits { max_parallel_requests: Some(1), ..KeyLimits::default() }
    )]
    #[case::a_limit_of_zero(json!({"rpm_limit": 0}), KeyLimits { rpm: Some(0), ..KeyLimits::default() })]
    #[case::per_model_in_its_metadata(
        json!({"metadata": {"model_rpm_limit": {"claude": 2}, "model_tpm_limit": {"claude": 100, "gpt": 0}}}),
        KeyLimits {
            model_rpm: ModelLimits::new(BTreeMap::from([("claude".to_owned(), 2)])),
            model_tpm: ModelLimits::new(BTreeMap::from([("claude".to_owned(), 100), ("gpt".to_owned(), 0)])),
            ..KeyLimits::default()
        }
    )]
    #[case::a_daily_token_limit_that_only_counts_batches(json!({"tpd_limit": 1}), KeyLimits::default())]
    #[tokio::test(flavor = "multi_thread")]
    async fn a_keys_limits_reach_its_caller(#[case] columns: Columns, #[case] limits: KeyLimits) {
        let caller = authenticate_holding(columns).await.unwrap();

        let Caller::Key(key) = &*caller else {
            panic!("the key authenticated as the master key");
        };
        assert_eq!(key.key.limits(), &limits);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn a_keys_metadata_priority_is_the_class_it_is_counted_under() {
        let caller = authenticate_holding(json!({"metadata": {"priority": "prod"}}))
            .await
            .unwrap();

        let Caller::Key(key) = &*caller else {
            panic!("the key authenticated as the master key");
        };
        assert_eq!(key.key.priority().map(|class| &**class), Some("prod"));
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

    /// A gateway with `redis` as its coordination Redis, and `enable_redis_auth_cache` set to
    /// `shared`.
    fn coordinated(redis: &RedisServer, shared: bool) -> Config {
        Config::from_yaml(&format!(
            "general_settings:\n  master_key: {MASTER_KEY}\n  coordination_redis:\n    host: 127.0.0.1\n    port: {}\nlitellm_settings:\n  enable_redis_auth_cache: {shared}\n",
            redis.port()
        ))
        .unwrap()
    }

    fn sharing(redis: &RedisServer) -> Config {
        coordinated(redis, true)
    }

    /// One gateway pod as `AppState` builds its auth for `config`, over the database `held`.
    async fn pod_for(config: &Config, held: &Held) -> Auth<NativeRows> {
        let resources = crate::Resources::build(config).unwrap();
        let clients = crate::redis::connect(config, &resources).await.unwrap();
        let keys = crate::state::keys(config, &clients, held.repositories());
        auth(config, Some(Arc::new(keys)))
    }

    /// One gateway pod as `AppState` builds its auth with `enable_redis_auth_cache` on: its own
    /// memory in front of `redis`, over the database `held`.
    async fn pod(redis: &RedisServer, held: &Held) -> Auth<NativeRows> {
        pod_for(&sharing(redis), held).await
    }

    fn stored_hash() -> String {
        Token::new(VIRTUAL_KEY).hash().to_string()
    }

    /// How long `redis` keeps its entry for [`VIRTUAL_KEY`], or a negative number with none.
    async fn seconds_cached(redis: &RedisServer) -> i64 {
        let ttl = redis.cli(&["TTL", &stored_hash()]).await;
        ttl.unwrap().trim().parse().unwrap()
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn a_key_read_from_the_database_is_cached_for_every_pod() {
        let redis = RedisServer::start().await.unwrap();
        let held = holding(json!({})).await;
        let pod_a = pod(&redis, &held).await;
        let pod_b = pod(&redis, &held).await;

        let first = bearing(&pod_a, VIRTUAL_KEY).await;
        let second = bearing(&pod_a, VIRTUAL_KEY).await;
        let other_pod = bearing(&pod_b, VIRTUAL_KEY).await;

        assert_eq!(
            (
                first.is_ok(),
                second.is_ok(),
                other_pod.is_ok(),
                held.key_reads(),
                (1..=60).contains(&seconds_cached(&redis).await),
            ),
            (true, true, true, 1, true)
        );
    }

    #[rstest]
    #[case::with_an_owner(json!({"user_id": "user"}))]
    #[case::in_a_team(json!({"team_id": "team"}))]
    #[case::with_a_budget(json!({"max_budget": 5.0}))]
    #[case::blocked(json!({"blocked": true}))]
    #[tokio::test(flavor = "multi_thread")]
    async fn a_key_python_answers_is_cached_for_python_by_the_read_that_declined(
        #[case] columns: Columns,
    ) {
        let redis = RedisServer::start().await.unwrap();
        let held = holding(columns).await;

        let authenticated = bearing(&pod(&redis, &held).await, VIRTUAL_KEY).await;

        assert_eq!(
            (authenticated.is_err(), seconds_cached(&redis).await > 0),
            (true, true)
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn without_the_shared_cache_a_key_is_kept_in_this_pods_memory_alone() {
        let redis = RedisServer::start().await.unwrap();
        let held = holding(json!({})).await;
        let gateway = pod_for(&coordinated(&redis, false), &held).await;

        let first = bearing(&gateway, VIRTUAL_KEY).await;
        let second = bearing(&gateway, VIRTUAL_KEY).await;

        assert_eq!(
            (
                first.is_ok(),
                second.is_ok(),
                held.key_reads(),
                seconds_cached(&redis).await
            ),
            (true, true, 1, -2)
        );
    }

    #[rstest]
    #[case::shared(true)]
    #[case::memory_alone(false)]
    #[tokio::test(flavor = "multi_thread")]
    async fn a_key_another_pod_blocks_stops_answering_from_this_pods_memory(#[case] shared: bool) {
        let redis = RedisServer::start().await.unwrap();
        let held = holding(json!({})).await;
        let config = coordinated(&redis, shared);
        let gateway = pod_for(&config, &held).await;
        let resources = crate::Resources::build(&config).unwrap();
        let clients = crate::redis::connect(&config, &resources).await.unwrap();
        let another_pod =
            crate::state::auth_cache(&config, &clients, litellm_redis_store::system_clock());
        let channel = CacheInvalidation::new(
            clients.coordination.clone().unwrap(),
            None,
            tokio::runtime::Handle::current(),
        );
        let served = bearing(&gateway, VIRTUAL_KEY).await.is_ok();

        held.execute(r#"UPDATE "LiteLLM_VerificationToken" SET blocked = true"#)
            .await;
        another_pod.delete(&[stored_hash()]).await.unwrap();

        let refused = eventually(async || {
            channel.broadcast_evictions(&[stored_hash()]).await;
            bearing(&gateway, VIRTUAL_KEY).await.err()
        })
        .await;
        assert!(served);
        assert!(matches!(refused, Rejected::InvalidToken), "{refused:?}");
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn a_key_read_from_redis_is_kept_in_memory_as_long_as_python_keeps_it() {
        let redis = RedisServer::start().await.unwrap();
        let config = sharing(&redis);
        let resources = crate::Resources::build(&config).unwrap();
        let clients = crate::redis::connect(&config, &resources).await.unwrap();
        let elapsed = Arc::new(std::sync::atomic::AtomicU64::new(0));
        let now = {
            let elapsed = elapsed.clone();
            Arc::new(move || Duration::from_secs(elapsed.load(std::sync::atomic::Ordering::SeqCst)))
        };
        let cache = crate::state::auth_cache(&config, &clients, now);
        let object = litellm_redis_store::auth_cache::AuthObject::Key(stored_hash());
        redis.cli(&["SET", &stored_hash(), "{}"]).await.unwrap();
        let read = async |seconds: u64| {
            elapsed.store(seconds, std::sync::atomic::Ordering::SeqCst);
            cache.get(None, &object).await.unwrap().is_some()
        };

        let from_redis = read(0).await;
        redis.cli(&["DEL", &stored_hash()]).await.unwrap();

        assert_eq!(
            (from_redis, read(59).await, read(61).await),
            (true, true, false)
        );
    }
}
