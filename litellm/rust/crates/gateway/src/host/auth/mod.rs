//! The gateway's answers to `gateway-auth-core`'s host steps on the key path, so it authenticates
//! a virtual key with no Python behind it, the native counterpart of python-bridge's
//! `PythonAuthHost`. A key's row, its team's and its project's are read and shaped by
//! `auth-store`, as the Python proxy's are, and `store.rs` puts the auth cache in front of them, as `UserApiKeyCache` sits in
//! front of Python's: this process's memory, and the Redis shared with the Python pods when
//! `enable_redis_auth_cache` attaches it, then the proxy's database on a miss, which fills it.
//!
//! A step this host does not answer yet declines, which sends the request to Python. A later
//! change serves more keys by answering one more step.

mod affiliation;
mod carried;
mod end_user;
mod key;
mod metadata;
mod models;
mod store;

use std::{convert::Infallible, sync::Arc, time::Duration};

use litellm_config::{Config, LitellmRole};
use litellm_gateway_auth::{Auth, BearerOrXApiKey, KeyAuthenticator, SystemClock};
use litellm_gateway_auth_core::{
    AffiliationRead, AgentIdentityFailure, AuthHost, Error, HostFacts, KeyHash, MasterKey,
    OperatorHooks, Refreshed, RouteGroups, Rows, Settings, Token,
    authn::{
        jwt::{KeyMappingRow, KeyMappingWrite, MappedClaim, MappedKey},
        key::Keyed,
    },
    body::Params,
    identity::{
        jwt::{
            AliasRead, DelegatedTeam, IdentityRead, ManagedAgent, ModelAccess, Org,
            RegisteredAgent, TeamRead, UserLookup, UserRead,
        },
        key::{KeyTeam, Owner},
    },
    record::{EndUserRecord, MembershipRecord, OrgRecord, ProjectRecord, TeamRecord, UserRecord},
};
use litellm_redis_store::auth_cache::KeyEntry;
use serde_json::Value;

pub(crate) use affiliation::{
    unserved, unserved_membership, unserved_org, unserved_project, unserved_user,
};
pub(crate) use end_user::{end_user, unserved as unserved_end_user};
pub use key::Key;
pub(crate) use models::{allowed_models, layers};
pub use store::AuthStore;

/// The rows the native host hands the core: a key's, and a core record for each thing a caller
/// belongs to. A key's team, owner, organization and project are read, and a caller that holds one still goes
/// to Python at admit. The steps that read the other records decline. `Infallible` is a row this
/// host has no record for, so no identity that needs one exists. The end user's record is empty
/// for a request that names none, and names the end user alone for one whose row holds nothing
/// the native app would have to apply.
#[derive(Clone, Debug, PartialEq)]
pub struct NativeRows;

impl Rows for NativeRows {
    type Key = Key;
    type User = UserRecord;
    type Team = TeamRecord;
    type EndUser = EndUserRecord;
    type ObjectPermission = Infallible;
    type Project = ProjectRecord;
    type Membership = MembershipRecord;
    type Org = OrgRecord;
    type Agent = Infallible;
}

pub struct NativeAuthHost {
    /// Where a virtual key is read from. `None` without a database, where the master key is the
    /// only caller.
    store: Option<Arc<AuthStore>>,
    /// The hash Python caches the master key's identity under, which is no key row.
    master: Option<KeyHash>,
    /// Whether a key's own budget is served: `BudgetCheck` reads its spend, which it does only
    /// with a database, and `general_settings.disable_budget_reservation` is on, so no call holds
    /// against it.
    own_budget: bool,
}

impl NativeAuthHost {
    /// The host the request body check reads stored credentials through, which reads no key.
    pub(crate) fn credential_store() -> Self {
        Self {
            store: None,
            master: None,
            own_budget: false,
        }
    }

    /// The store, or `step` declined without a database to read.
    fn store(&self, step: &'static str) -> Result<&AuthStore, Error> {
        self.store.as_deref().map_or_else(|| declined(step), Ok)
    }
}

/// The request needs `step`, which this host does not answer yet.
fn declined<T>(step: &'static str) -> Result<T, Error> {
    tracing::debug!(
        step,
        "the native auth host does not serve what this request needs"
    );
    Err(Error::HostDeclined)
}

/// A read the database failed is declined, so the request goes to Python.
fn unless_failed<T>(
    read: Result<T, litellm_auth_store::Error>,
    step: &'static str,
) -> Result<T, Error> {
    read.or_else(|error| {
        tracing::warn!(
            error = %litellm_error::Report::new(&error),
            step,
            "the native auth host could not read what this request needs"
        );
        declined(step)
    })
}

fn keyed(key: &KeyEntry) -> Result<Option<Keyed<Key>>, Error> {
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
/// for `config`, the master key and with `store` a virtual key read through it, taken from the
/// header an inference client sends its key in. `budgets_checked` is whether the budget checks
/// are registered, which read a counter's spend from the database.
pub fn auth(
    config: &Config,
    store: Option<Arc<AuthStore>>,
    budgets_checked: bool,
) -> Auth<NativeRows> {
    let settings = settings(config, store.is_some());
    let host = NativeAuthHost {
        store,
        master: master_key(config).map(|key| Token::new(key.clone()).hash()),
        own_budget: budgets_checked && config.general_settings.disable_budget_reservation,
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
        let Some(store) = &self.store else {
            return Ok(None);
        };
        match Box::pin(store.cached(hash)).await {
            Some(key) => keyed(&key),
            None => Ok(None),
        }
    }

    /// The core asks here only after `cached_key` missed, so the cache is not read again. The
    /// store caches the key it found, whoever answers the request.
    async fn db_key(&self, hash: &KeyHash, _: &EndUserRecord) -> Result<Option<Keyed<Key>>, Error> {
        let store = self.store("db_key")?;
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

    async fn managed_agent(&self, _: &Value) -> Result<Option<ManagedAgent<Infallible>>, Error> {
        declined("managed_agent")
    }

    async fn registered_agent(&self, _: &str) -> Result<Option<RegisteredAgent>, Error> {
        declined("registered_agent")
    }

    async fn delegated_agent_team(&self, _: &DelegatedTeam<'_>) -> Result<Option<String>, Error> {
        declined("delegated_agent_team")
    }

    /// A user id with no row is no owner, like the key path's `get_user_object`, which swallows
    /// `UserNotFoundError`.
    async fn key_owner(&self, user_id: &str) -> Result<Option<Owner<UserRecord>>, Error> {
        let read = self.store("key_owner")?.user(user_id).await;
        let Some(user) = unless_failed(read, "key_owner")? else {
            return Ok(None);
        };
        affiliation::user(&user)
            .map(|user| {
                Some(Owner {
                    deactivated: user.deactivated,
                    row: user.row,
                })
            })
            .or_else(declined)
    }

    async fn user(&self, _: &UserLookup<'_>) -> Result<UserRead<UserRecord>, Error> {
        declined("user")
    }

    async fn resolve_user_identity(
        &self,
        _: &UserLookup<'_>,
        _: Option<&str>,
    ) -> Result<IdentityRead<NativeRows>, Error> {
        declined("resolve_user_identity")
    }

    /// The core asks for the cached team only to bring a key's own copy of the team's values up
    /// to date. This host reads none of that copy, since [`Self::key_team`] reads the team's row,
    /// so it has nothing to refresh a key from.
    async fn cached_team(&self, _: &str) -> Result<Option<Refreshed<TeamRecord>>, Error> {
        Ok(None)
    }

    /// A team with no row declines, since the key is then judged by its own copy of the team's
    /// values, which this host does not read.
    async fn key_team(&self, team_id: &str) -> Result<Option<KeyTeam<TeamRecord>>, Error> {
        let read = self.store("key_team")?.team(team_id).await;
        match unless_failed(read, "key_team")? {
            Some(team) => affiliation::team(&team)
                .map(|team| {
                    Some(KeyTeam {
                        organization_id: team.row.organization_id.clone(),
                        row: team.row,
                    })
                })
                .or_else(declined),
            None => declined("team_id"),
        }
    }

    async fn team(&self, _: &str, _: bool) -> Result<TeamRead<TeamRecord>, Error> {
        declined("team")
    }

    async fn granting_team(&self, _: &str) -> Result<TeamRead<TeamRecord>, Error> {
        declined("granting_team")
    }

    async fn team_by_alias(&self, _: &str) -> Result<AliasRead<TeamRecord>, Error> {
        declined("team_by_alias")
    }

    async fn load_team_membership(
        &self,
        user_id: &str,
        team_id: &str,
        read: AffiliationRead,
    ) -> Result<Option<MembershipRecord>, Error> {
        let store = self.store("load_team_membership")?;
        let membership = store.team_membership(user_id, team_id).await;
        match unless_failed(membership, "load_team_membership")? {
            Some(membership) => affiliation::membership(&membership, read)
                .map(Some)
                .or_else(declined),
            None => Ok(None),
        }
    }

    async fn org(&self, id: &str, read: AffiliationRead) -> Result<Option<Org<OrgRecord>>, Error> {
        if read == AffiliationRead::Jwt {
            return declined("org");
        }
        let read = Box::pin(self.store("org")?.org(id)).await;
        unless_failed(read, "org")?
            .map(|row| affiliation::org(&row).or_else(declined))
            .transpose()
    }

    async fn org_by_alias(&self, _: &str) -> Result<Option<Org<OrgRecord>>, Error> {
        declined("org_by_alias")
    }

    async fn jwt_end_user(&self, _: &str) -> Result<EndUserRecord, Error> {
        declined("jwt_end_user")
    }

    /// A request that names no end user reads none. The gateway reads the body after it
    /// authenticates, so it hands the core no end user, and `admit` reads the one a body names
    /// through [`end_user`] once the body is read.
    async fn end_user(&self, end_user_id: Option<&str>) -> Result<EndUserRecord, Error> {
        match end_user_id {
            None => Ok(EndUserRecord::default()),
            Some(_) => declined("end_user"),
        }
    }

    async fn validated_end_user_id(&self, _: &str) -> Result<Option<String>, Error> {
        declined("validated_end_user_id")
    }

    async fn object_permission(&self, _: &str) -> Result<Option<Infallible>, Error> {
        declined("object_permission")
    }

    /// A project with no row declines, since a key can name one only once its row is deleted,
    /// which this host does not judge the key after.
    async fn project(&self, project_id: &str) -> Result<Option<ProjectRecord>, Error> {
        let read = self.store("project")?.project(project_id).await;
        match unless_failed(read, "project")? {
            Some(project) => affiliation::project(&project).map(Some).or_else(declined),
            None => declined("project_id"),
        }
    }

    /// The gateway reads each object when the core asks for it, with nothing to warm.
    async fn prefetch_identity(&self, _: Option<&Token>, _: Option<&str>) -> Result<(), Error> {
        Ok(())
    }

    async fn prefetch_auth_objects(&self, _: &Key, _: &EndUserRecord) -> Result<(), Error> {
        Ok(())
    }

    async fn set_user_role(&self, _: &UserRecord, _: LitellmRole) -> Result<UserRecord, Error> {
        declined("set_user_role")
    }

    async fn patch_team_membership(
        &self,
        _: &UserRecord,
        _: &[String],
        _: &[String],
    ) -> Result<UserRecord, Error> {
        declined("patch_team_membership")
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

    async fn team_model_access(&self, _: &TeamRecord, _: &str) -> Result<ModelAccess, Error> {
        declined("team_model_access")
    }

    async fn team_member_budget(&self, _: &Key, _: Option<&UserRecord>) -> Result<(), Error> {
        Ok(())
    }

    /// A key's own budget is enforced by `BudgetCheck` at read time, so it is served while the
    /// check is registered and the budget reservation is disabled.
    // TODO(#54354): serve it with the reservation on too once the gateway reserves against a budget
    // before the call, as Python does with it on
    async fn key_budgets(&self, key: &Key, _: Option<&UserRecord>, _: bool) -> Result<(), Error> {
        match key
            .budget
            .or(key.checked_budget.filter(|_| !self.own_budget))
        {
            Some(column) => declined(column),
            None => Ok(()),
        }
    }

    async fn jwt_user_model_budget(
        &self,
        _: &UserRecord,
        _: &str,
        _: Option<&str>,
    ) -> Result<(), Error> {
        declined("jwt_user_model_budget")
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

    async fn with_team(&self, _: &Key, _: &TeamRecord) -> Result<Key, Error> {
        declined("with_team")
    }

    // TODO: declined until the gateway reads the credentials table, which the native request body
    // check needs once a gateway route runs it
    async fn credential(&self, _: &str) -> Result<Option<Params>, Error> {
        declined("credential")
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
    use litellm_gateway_auth::{Authenticated, Error as Rejected, RequireMasterKey};
    use litellm_gateway_auth_core::{
        Credential, Holder, Named, Team,
        authz::{Layer, ModelList},
    };
    use litellm_redis_store::{
        auth_cache::AuthObject, cache_invalidation::CacheInvalidation,
        spend_counters::SpendCounters,
    };
    use litellm_redis_testing::{RedisServer, eventually};
    use litellm_repositories::Database;
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
            VALUES ('org', 'Org', 'budget', 'test', 'test'),
                   ('other', 'Other Org', 'budget', 'test', 'test');
        INSERT INTO "LiteLLM_TeamTable" (team_id) VALUES ('team');
        INSERT INTO "LiteLLM_ProjectTable" (project_id, project_alias, team_id, models, created_by, updated_by)
            VALUES ('project', 'Project', 'team', ARRAY[]::text[], 'test', 'test');
        INSERT INTO "LiteLLM_ObjectPermissionTable" (object_permission_id) VALUES ('permission');
        INSERT INTO "LiteLLM_ModelTable" (id, created_by, updated_by) VALUES (7, 'test', 'test');
    "#;

    const SQUAD: &str = "squad";

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
    fn memory_only(database: Database) -> Arc<AuthStore> {
        Arc::new(AuthStore::new(keys::memory_cache(), database))
    }

    async fn authenticate(
        config: &Config,
        keys: Option<Database>,
        header: (&str, &str),
    ) -> Result<Authenticated<NativeRows>, Rejected> {
        auth(config, keys.map(memory_only), true)
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
        database: Database,
        token: &str,
    ) -> Result<Authenticated<NativeRows>, Rejected> {
        bearing(
            &auth(&with_master_key(), Some(memory_only(database)), true),
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

    async fn in_team(key: Columns, team: Columns) -> Held {
        let held = Held::start().await;
        held.execute(PARENTS).await;
        held.key(VIRTUAL_KEY, merged(json!({"team_id": SQUAD}), key))
            .await;
        held.insert_named(
            "LiteLLM_TeamTable",
            &merged(json!({"team_id": SQUAD}), team),
        )
        .await;
        held
    }

    const ATLAS: &str = "atlas";

    async fn in_project(project: Columns) -> Held {
        let held = Held::start().await;
        held.execute(PARENTS).await;
        held.insert_named(
            "LiteLLM_ProjectTable",
            &merged(
                json!({"project_id": ATLAS, "created_by": "test", "updated_by": "test"}),
                project,
            ),
        )
        .await;
        held.key(VIRTUAL_KEY, json!({"project_id": ATLAS})).await;
        held
    }

    fn merged(base: Columns, over: Columns) -> Columns {
        let fields = base.as_object().into_iter().chain(over.as_object());
        Columns::Object(
            fields
                .flatten()
                .map(|(name, value)| (name.clone(), value.clone()))
                .collect(),
        )
    }

    #[rstest]
    #[case::without_a_database(None)]
    #[case::with_a_database_that_is_never_read(Some(keys::unreachable()))]
    #[tokio::test]
    async fn the_master_key_is_the_admin_caller_with_and_without_a_database(
        #[case] keys: Option<Database>,
        #[values(("authorization", "Bearer sk-master-credential"), ("x-api-key", MASTER_KEY))]
        header: (&str, &str),
    ) {
        let caller = authenticate(&with_master_key(), keys, header)
            .await
            .unwrap();

        assert_eq!(
            (&caller.credential, caller.role()),
            (&Credential::MasterKey, LitellmRole::ProxyAdmin)
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
                &config, None, false,
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

        let Credential::Key { key, .. } = &caller.credential else {
            panic!("the key authenticated as the master key");
        };
        assert_eq!(
            (
                caller.hash(),
                caller.role(),
                key.record.expires,
                allowed_models(&caller),
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
    #[case::a_team_with_no_row(json!({"team_id": "gone"}))]
    #[case::the_ui_team(json!({"team_id": "litellm-dashboard"}))]
    #[case::agent_id(json!({"agent_id": "agent"}))]
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
    #[case::max_budget(json!({"max_budget": 5.0}), true)]
    #[case::max_budget_spent(json!({"max_budget": 5.0, "spend": 7.0}), true)]
    #[case::a_reset_schedule(
        json!({"max_budget": 5.0, "budget_duration": "30d", "budget_reset_at": FAR_FUTURE}),
        true
    )]
    #[case::soft_budget_cooldown(json!({"max_budget": 5.0, "soft_budget_cooldown": true}), false)]
    #[case::budget_limits(json!({"max_budget": 5.0, "budget_limits": [{"a": 1}]}), false)]
    #[case::budget_fallbacks(json!({"max_budget": 5.0, "budget_fallbacks": {"a": ["b"]}}), false)]
    #[case::budget_id(json!({"max_budget": 5.0, "budget_id": "budget"}), false)]
    #[case::model_max_budget(json!({"max_budget": 5.0, "model_max_budget": {"claude": 1}}), false)]
    #[case::a_temporary_increase(
        json!({"max_budget": 5.0, "metadata": {"temp_budget_increase": 5.0, "temp_budget_expiry": FAR_FUTURE}}),
        false
    )]
    #[case::throttle_on_budget_exceeded(
        json!({"max_budget": 5.0, "rpm_limit": 2, "metadata": {"throttle_on_budget_exceeded": true}}),
        false
    )]
    #[tokio::test(flavor = "multi_thread")]
    async fn a_keys_own_budget_is_served_only_while_the_reservation_is_disabled(
        #[case] columns: Columns,
        #[case] served_without_reservation: bool,
        #[values(true, false)] reservation_disabled: bool,
    ) {
        let held = holding(columns).await;
        let config = config(&format!(
            "  master_key: {MASTER_KEY}\n  disable_budget_reservation: {reservation_disabled}\n"
        ));

        let authenticated = bearing(
            &auth(&config, Some(memory_only(held.database())), true),
            VIRTUAL_KEY,
        )
        .await;

        let budget = match authenticated {
            Ok(caller) => match &caller.credential {
                Credential::Key { key, .. } => Some(key.budget().max),
                Credential::MasterKey | Credential::Jwt { .. } => {
                    panic!("the key authenticated as something else")
                }
            },
            Err(Rejected::NotNative) => None,
            Err(other) => panic!("{other:?}"),
        };
        assert_eq!(
            budget,
            (served_without_reservation && reservation_disabled).then_some(Some(5.0))
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn a_keys_own_budget_is_not_native_while_no_check_reads_its_spend() {
        let held = holding(json!({"max_budget": 5.0})).await;
        let config = config(&format!(
            "  master_key: {MASTER_KEY}\n  disable_budget_reservation: true\n"
        ));

        let refused = bearing(
            &auth(&config, Some(memory_only(held.database())), false),
            VIRTUAL_KEY,
        )
        .await;

        assert!(
            matches!(refused, Err(Rejected::NotNative)),
            "{:?}",
            refused.err()
        );
    }

    #[rstest]
    #[case::org_id(json!({"organization_id": "org"}), json!({}))]
    #[case::organization_id(json!({}), json!({"organization_id": "org"}))]
    #[case::key_org_wins(json!({"organization_id": "org"}), json!({"organization_id": "other"}))]
    #[tokio::test(flavor = "multi_thread")]
    async fn a_keys_organization_reaches_its_caller(#[case] key: Columns, #[case] team: Columns) {
        let held = in_team(key, team).await;

        let caller = presenting(held.database(), VIRTUAL_KEY).await.unwrap();

        assert_eq!(
            caller
                .affiliation
                .org
                .as_ref()
                .and_then(Named::stored)
                .map(|record| (
                    record.organization_id.as_str(),
                    record.organization_alias.as_deref()
                )),
            Some(("org", Some("Org"))),
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

        let Credential::Key { key, .. } = &caller.credential else {
            panic!("the key authenticated as the master key");
        };
        assert_eq!(key.limits(), &limits);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn a_keys_metadata_priority_is_the_class_it_is_counted_under() {
        let caller = authenticate_holding(json!({"metadata": {"priority": "prod"}}))
            .await
            .unwrap();

        let Credential::Key { key, .. } = &caller.credential else {
            panic!("the key authenticated as the master key");
        };
        assert_eq!(key.priority().map(|class| &**class), Some("prod"));
    }

    #[rstest]
    #[tokio::test(flavor = "multi_thread")]
    async fn a_key_in_a_team_holds_the_teams_row_and_is_checked_against_both_lists() {
        let held = in_team(
            json!({"models": ["haiku"]}),
            json!({"team_alias": "Squad", "models": ["claude"]}),
        )
        .await;

        let caller = presenting(held.database(), VIRTUAL_KEY).await.unwrap();

        let (key, team) = (["haiku".to_owned()], ["claude".to_owned()]);
        assert_eq!(
            (
                &caller.affiliation.holder,
                layers(&caller).collect::<Vec<_>>(),
                allowed_models(&caller),
            ),
            (
                &Holder::Team(Team::Row(TeamRecord {
                    team_id: SQUAD.to_owned(),
                    team_alias: Some("Squad".to_owned()),
                    models: team.to_vec(),
                    ..TeamRecord::default()
                })),
                vec![
                    Layer::Key(ModelList::read(&key)),
                    Layer::Team(ModelList::read(&team)),
                ],
                [key.clone(), team.clone()].concat(),
            )
        );
    }

    #[rstest]
    #[case::object_permission_id(json!({"object_permission_id": "permission"}))]
    #[case::model_aliases(json!({"model_id": 7}))]
    #[case::metadata(json!({"metadata": {"guardrails": ["pii"]}}))]
    #[case::metadata_beside_a_model_limit(json!({"metadata": {"model_rpm_limit": {"claude": 2}, "tags": ["a"]}}))]
    #[case::a_priority_that_is_no_text(json!({"metadata": {"priority": 5}}))]
    #[case::budget_reset_at(json!({"budget_reset_at": "1970-01-01T00:00:09"}))]
    #[case::budget_limits(json!({"budget_limits": [{"a": 1}]}))]
    #[case::model_max_budget(json!({"model_max_budget": {"claude": 1}}))]
    #[case::router_settings(json!({"router_settings": {"a": 1}}))]
    #[case::a_model_pattern(json!({"models": ["claude", "anthropic/*"]}))]
    #[case::the_team_models(json!({"models": ["all-team-models"]}))]
    #[tokio::test(flavor = "multi_thread")]
    async fn a_key_whose_team_holds_what_no_record_field_has_a_type_for_is_not_native(
        #[case] team: Columns,
    ) {
        let held = in_team(json!({}), team).await;

        let refused = presenting(held.database(), VIRTUAL_KEY).await;

        assert!(
            matches!(refused, Err(Rejected::NotNative)),
            "{:?}",
            refused.err()
        );
    }

    #[rstest]
    #[case::max_budget(json!({"max_budget": 5.0}), "max_budget")]
    #[case::soft_budget(json!({"soft_budget": 1.0}), "soft_budget")]
    #[case::budget_duration(json!({"budget_duration": "30d"}), "budget_duration")]
    #[case::tpm_limit(json!({"tpm_limit": 100}), "tpm_limit")]
    #[case::rpm_limit(json!({"rpm_limit": 10}), "rpm_limit")]
    #[case::a_limit_of_zero(json!({"rpm_limit": 0}), "rpm_limit")]
    #[case::tpd_limit(json!({"tpd_limit": 100}), "tpd_limit")]
    #[case::max_parallel_requests(json!({"max_parallel_requests": 1}), "max_parallel_requests")]
    #[case::blocked(json!({"blocked": true}), "blocked")]
    #[case::access_group_ids(json!({"access_group_ids": ["group"]}), "access_group_ids")]
    #[case::policies(json!({"policies": ["policy"]}), "policies")]
    #[case::default_team_member_models(
        json!({"default_team_member_models": ["claude"]}),
        "default_team_member_models"
    )]
    #[case::allow_team_guardrail_config(
        json!({"allow_team_guardrail_config": true}),
        "allow_team_guardrail_config"
    )]
    #[case::model_rpm_limit(json!({"metadata": {"model_rpm_limit": {"claude": 2}}}), "model_rpm_limit")]
    #[case::model_tpm_limit(json!({"metadata": {"model_tpm_limit": {"claude": 100}}}), "model_tpm_limit")]
    #[case::priority(json!({"metadata": {"priority": "prod"}}), "priority")]
    #[tokio::test(flavor = "multi_thread")]
    async fn a_key_in_a_team_copies_each_plain_column_its_team_holds_and_names_it_unserved(
        #[case] team: Columns,
        #[case] column: &str,
    ) {
        let held = in_team(json!({}), team).await;

        let caller = presenting(held.database(), VIRTUAL_KEY).await.unwrap();

        let Some(Team::Row(team)) = caller.affiliation.holder.team() else {
            panic!("the key holds no team row");
        };
        assert_eq!(unserved(team), Some(column));
    }

    /// The database holding [`VIRTUAL_KEY`], owned by a user whose row has `owner` set on it.
    async fn owned_by(owner: Columns) -> Held {
        let held = Held::start().await;
        held.execute(PARENTS).await;
        held.key(VIRTUAL_KEY, json!({"user_id": "ada"})).await;
        held.insert_named(
            "LiteLLM_UserTable",
            &merged(json!({"user_id": "ada"}), owner),
        )
        .await;
        held
    }

    #[rstest]
    #[case::team_id(json!({"team_id": "squad"}), "team_id")]
    #[case::organization_id(json!({"organization_id": "org"}), "organization_id")]
    #[case::models(json!({"models": ["claude"]}), "models")]
    #[case::tpm_limit(json!({"tpm_limit": 100}), "tpm_limit")]
    #[case::rpm_limit(json!({"rpm_limit": 10}), "rpm_limit")]
    #[case::a_limit_of_zero(json!({"rpm_limit": 0}), "rpm_limit")]
    #[case::max_parallel_requests(json!({"max_parallel_requests": 1}), "max_parallel_requests")]
    #[case::max_budget(json!({"max_budget": 5.0}), "max_budget")]
    #[case::rollover_max_budget(json!({"rollover_max_budget": 5.0}), "rollover_max_budget")]
    #[case::rollover_credit(json!({"rollover_credit": 1.5}), "rollover_credit")]
    #[case::budget_duration(json!({"budget_duration": "30d"}), "budget_duration")]
    #[case::allowed_cache_controls(
        json!({"allowed_cache_controls": ["no-cache"]}),
        "allowed_cache_controls"
    )]
    #[case::policies(json!({"policies": ["policy"]}), "policies")]
    #[tokio::test(flavor = "multi_thread")]
    async fn a_key_copies_each_plain_column_its_owner_holds_and_names_it_unserved(
        #[case] owner: Columns,
        #[case] column: &str,
    ) {
        let held = owned_by(owner).await;

        let caller = presenting(held.database(), VIRTUAL_KEY).await.unwrap();

        let Some(Named::Stored(owner)) = caller.affiliation.holder.user() else {
            panic!("the key holds no owner row");
        };
        assert_eq!(unserved_user(owner), Some(column));
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn an_owners_email_role_and_teams_are_copied_and_keep_no_caller_with_python() {
        let described = json!({
            "user_email": "ada@example.com",
            "user_role": "internal_user",
            "teams": ["squad"],
        });
        let held = owned_by(described).await;

        let caller = presenting(held.database(), VIRTUAL_KEY).await.unwrap();

        let Some(Named::Stored(owner)) = caller.affiliation.holder.user() else {
            panic!("the key holds no owner row");
        };
        assert_eq!(
            (
                owner.user_email.as_deref(),
                owner.user_role.as_deref(),
                owner.teams.as_slice(),
                unserved_user(owner),
            ),
            (
                Some("ada@example.com"),
                Some("internal_user"),
                &["squad".to_owned()][..],
                None,
            )
        );
    }

    #[rstest]
    #[tokio::test(flavor = "multi_thread")]
    async fn a_team_is_read_from_the_database_again_only_once_it_is_evicted() {
        let held = in_team(json!({}), json!({})).await;
        let cache = keys::memory_cache();
        let store = Arc::new(AuthStore::new(cache.clone(), held.database()));
        let gateway = auth(&with_master_key(), Some(store), false);

        let budget = async || {
            let caller = bearing(&gateway, VIRTUAL_KEY).await.unwrap();
            match caller.affiliation.holder.team() {
                Some(Team::Row(team)) => team.budget.max,
                _ => None,
            }
        };

        let read = budget().await;
        held.execute(r#"UPDATE "LiteLLM_TeamTable" SET max_budget = 1 WHERE team_id = 'squad'"#)
            .await;
        let kept = budget().await;
        cache
            .delete_local(&AuthObject::Team(SQUAD.to_owned()).key())
            .unwrap();
        let read_again = budget().await;

        assert_eq!((read, kept, read_again), (None, None, Some(1.0)));
    }

    #[rstest]
    #[tokio::test(flavor = "multi_thread")]
    async fn a_team_read_from_the_database_is_cached_for_python_with_the_time_it_was_read() {
        let redis = RedisServer::start().await.unwrap();
        let held = in_team(json!({}), json!({})).await;
        let key = AuthObject::Team(SQUAD.to_owned()).key();

        let authenticated = bearing(&pod(&redis, &held).await, VIRTUAL_KEY).await;

        let cached: Columns =
            serde_json::from_str(&redis.cli(&["GET", &key]).await.unwrap()).unwrap();
        let ttl = redis.cli(&["TTL", &key]).await.unwrap();
        let seconds: i64 = ttl.trim().parse().unwrap();
        assert!(authenticated.is_ok());
        assert_eq!(
            (
                &cached["team_id"],
                cached["last_refreshed_at"]
                    .as_f64()
                    .is_some_and(|at| at > 0.0),
                (1..=60).contains(&seconds),
            ),
            (&json!(SQUAD), true, true)
        );
    }

    #[rstest]
    #[case::team_id(json!({"team_id": "team"}), "team_id")]
    #[case::models(json!({"models": ["claude"]}), "models")]
    #[case::blocked(json!({"blocked": true}), "blocked")]
    #[tokio::test(flavor = "multi_thread")]
    async fn a_key_in_a_project_copies_each_plain_column_its_project_holds_and_names_it_unserved(
        #[case] project: Columns,
        #[case] column: &str,
    ) {
        let held = in_project(project).await;

        let caller = presenting(held.database(), VIRTUAL_KEY).await.unwrap();

        let Some(Named::Stored(project)) = &caller.affiliation.project else {
            panic!("the key holds no project row");
        };
        assert_eq!(unserved_project(project), Some(column));
    }

    #[rstest]
    #[case::budget_id(json!({"budget_id": "budget"}))]
    #[case::metadata(json!({"metadata": {"tags": ["a"]}}))]
    #[case::model_rpm_limit(json!({"model_rpm_limit": {"claude": 1}}))]
    #[case::model_tpm_limit(json!({"model_tpm_limit": {"claude": 1}}))]
    #[case::object_permission_id(json!({"object_permission_id": "permission"}))]
    #[tokio::test(flavor = "multi_thread")]
    async fn a_key_whose_project_holds_what_no_record_field_has_a_type_for_is_not_native(
        #[case] project: Columns,
    ) {
        let held = in_project(project).await;

        let refused = presenting(held.database(), VIRTUAL_KEY).await;

        assert!(
            matches!(refused, Err(Rejected::NotNative)),
            "{:?}",
            refused.err()
        );
    }

    #[rstest]
    #[tokio::test(flavor = "multi_thread")]
    async fn a_project_is_read_from_the_database_again_only_once_it_is_evicted() {
        let held = in_project(json!({})).await;
        let cache = keys::memory_cache();
        let store = Arc::new(AuthStore::new(cache.clone(), held.database()));
        let gateway = auth(&with_master_key(), Some(store), false);

        let blocked = async || {
            let caller = bearing(&gateway, VIRTUAL_KEY).await.unwrap();
            let project = caller.affiliation.project.as_ref();
            project
                .and_then(Named::stored)
                .map(|project| project.blocked)
        };

        let read = blocked().await;
        held.execute(
            r#"UPDATE "LiteLLM_ProjectTable" SET blocked = true WHERE project_id = 'atlas'"#,
        )
        .await;
        let kept = blocked().await;
        cache
            .delete_local(&AuthObject::Project(ATLAS.to_owned()).key())
            .unwrap();
        let read_again = blocked().await;

        assert_eq!(
            (read, kept, read_again),
            (Some(false), Some(false), Some(true))
        );
    }

    #[rstest]
    #[tokio::test(flavor = "multi_thread")]
    async fn a_project_read_from_the_database_is_cached_for_python_with_the_time_it_was_read() {
        let redis = RedisServer::start().await.unwrap();
        let held = in_project(json!({})).await;
        let key = AuthObject::Project(ATLAS.to_owned()).key();

        let authenticated = bearing(&pod(&redis, &held).await, VIRTUAL_KEY).await;

        let cached: Columns =
            serde_json::from_str(&redis.cli(&["GET", &key]).await.unwrap()).unwrap();
        let ttl = redis.cli(&["TTL", &key]).await.unwrap();
        let seconds: i64 = ttl.trim().parse().unwrap();
        assert!(authenticated.is_ok());
        assert_eq!(
            (
                &cached["project_id"],
                cached["last_refreshed_at"]
                    .as_f64()
                    .is_some_and(|at| at > 0.0),
                (1..=60).contains(&seconds),
            ),
            (&json!(ATLAS), true, true)
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
            caller
                .ok()
                .map(|caller| caller.credential == Credential::MasterKey),
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
        let store = crate::state::auth_store(
            config,
            &clients,
            held.database(),
            Arc::new(SpendCounters::new(None)),
        );
        auth(config, Some(Arc::new(store)), true)
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
    #[case::with_a_soft_budget_cooldown(json!({"soft_budget_cooldown": true}))]
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

    /// The clock a test moves, at the seconds it was last set to.
    fn frozen() -> (
        litellm_redis_store::Clock,
        Arc<std::sync::atomic::AtomicU64>,
    ) {
        let elapsed = Arc::new(std::sync::atomic::AtomicU64::new(0));
        let read = elapsed.clone();
        let now =
            Arc::new(move || Duration::from_secs(read.load(std::sync::atomic::Ordering::SeqCst)));
        (now, elapsed)
    }

    /// One gateway pod with no coordination Redis, its auth cache built as `AppState` builds it for
    /// `general_settings`, over the database `held`, on the clock `now`.
    fn pod_on_clock(
        general_settings: &str,
        held: &Held,
        now: litellm_redis_store::Clock,
    ) -> Auth<NativeRows> {
        let config = config(&format!("  master_key: {MASTER_KEY}\n{general_settings}"));
        let cache = crate::state::auth_cache(&config, &crate::RedisClients::default(), now);
        auth(
            &config,
            Some(Arc::new(AuthStore::new(Arc::new(cache), held.database()))),
            true,
        )
    }

    #[rstest]
    #[case::unset("", 59, 61)]
    #[case::configured("  user_api_key_cache_ttl: 600\n", 599, 601)]
    #[case::configured_as_text("  user_api_key_cache_ttl: \"30\"\n", 29, 31)]
    #[tokio::test(flavor = "multi_thread")]
    async fn a_key_is_read_from_the_database_again_once_the_cache_ttl_passes(
        #[case] general_settings: &str,
        #[case] still_cached: u64,
        #[case] expired: u64,
    ) {
        let held = holding(json!({})).await;
        let (now, elapsed) = frozen();
        let gateway = pod_on_clock(general_settings, &held, now);
        let read_at = async |seconds: u64| {
            elapsed.store(seconds, std::sync::atomic::Ordering::SeqCst);
            assert!(bearing(&gateway, VIRTUAL_KEY).await.is_ok());
            held.key_reads()
        };

        let reads = (
            read_at(0).await,
            read_at(still_cached).await,
            read_at(expired).await,
        );

        assert_eq!(reads, (1, 1, 2));
    }

    #[rstest]
    #[case::unset("", 1..=60)]
    #[case::configured("  user_api_key_cache_ttl: 600\n", 541..=600)]
    #[tokio::test(flavor = "multi_thread")]
    async fn the_shared_cache_keeps_a_key_for_the_cache_ttl(
        #[case] general_settings: &str,
        #[case] expected: std::ops::RangeInclusive<i64>,
    ) {
        let redis = RedisServer::start().await.unwrap();
        let held = holding(json!({})).await;
        let config = Config::from_yaml(&format!(
            "general_settings:\n  master_key: {MASTER_KEY}\n  coordination_redis:\n    host: 127.0.0.1\n    port: {}\n{general_settings}litellm_settings:\n  enable_redis_auth_cache: true\n",
            redis.port()
        ))
        .unwrap();

        let authenticated = bearing(&pod_for(&config, &held).await, VIRTUAL_KEY).await;

        let seconds = seconds_cached(&redis).await;
        assert!(authenticated.is_ok());
        assert!(expected.contains(&seconds), "{seconds}");
    }

    #[rstest]
    #[case::full_at_two("  user_api_key_cache_max_size: 2\n", (3, 4))]
    #[case::unset("", (3, 3))]
    #[tokio::test(flavor = "multi_thread")]
    async fn a_full_cache_evicts_the_key_soonest_to_expire(
        #[case] general_settings: &str,
        #[case] expected: (usize, usize),
    ) {
        let held = Held::start().await;
        held.key("sk-first", json!({})).await;
        held.key("sk-second", json!({})).await;
        held.key("sk-third", json!({})).await;
        let (now, elapsed) = frozen();
        let gateway = pod_on_clock(general_settings, &held, now);
        let read_at = async |token: &str, seconds: u64| {
            elapsed.store(seconds, std::sync::atomic::Ordering::SeqCst);
            assert!(bearing(&gateway, token).await.is_ok());
            held.key_reads()
        };
        read_at("sk-first", 0).await;
        read_at("sk-second", 1).await;
        read_at("sk-third", 2).await;

        let reads = (read_at("sk-third", 2).await, read_at("sk-first", 2).await);

        assert_eq!(reads, expected);
    }
}
