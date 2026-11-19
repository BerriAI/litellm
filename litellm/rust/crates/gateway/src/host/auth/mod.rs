//! The gateway's answers to `gateway-auth-core`'s host steps on the key path, so it authenticates
//! a virtual key with no Python behind it, the native counterpart of python-bridge's
//! `PythonAuthHost`. A key is read from Python's Redis auth cache when the gateway shares one,
//! and from the proxy's database on a miss, which then fills that cache. With no shared cache it
//! is read from the database on every request.
//!
//! A step this host does not answer yet declines, which sends the request to Python. A later
//! change serves more keys by answering one more step.

#[macro_use]
mod key;
mod cached;

use std::{convert::Infallible, sync::Arc, time::Duration};

use futures_util::future::BoxFuture;
use litellm_config::{Config, LitellmRole};
use litellm_error::Source;
use litellm_gateway_auth::{Authenticator, KeyAuthenticator};
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
use litellm_redis_store::auth_cache::AuthCache;
use litellm_repositories::{KeyRepository, KeyView, Side};
use serde_json::Value;

pub use key::Key;

pub type RowFuture<'a> = BoxFuture<'a, Result<Option<KeyView>, Source>>;

/// Where the native host reads a virtual key's row: the proxy's database, or the rows a test
/// holds.
pub trait KeyRows: Send + Sync {
    fn find<'a>(&'a self, hash: &'a str) -> RowFuture<'a>;
}

/// Rows held in memory, each under its key's hash.
impl KeyRows for std::collections::BTreeMap<String, KeyView> {
    fn find<'a>(&'a self, hash: &'a str) -> RowFuture<'a> {
        Box::pin(async move { Ok(self.get(hash).cloned()) })
    }
}

impl KeyRows for KeyRepository {
    fn find<'a>(&'a self, hash: &'a str) -> RowFuture<'a> {
        Box::pin(async move {
            let found = KeyRepository::find(self, hash, Side::Read)
                .await
                .map_err(Source::new)?;
            Ok(found.map(|key| key.view().clone()))
        })
    }
}

/// The rows the native host hands the core: a key's, and none of the objects it does not read
/// yet.
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
    rows: Arc<dyn KeyRows>,
    cache: Option<AuthCache>,
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

fn keyed(row: &KeyView, also: Option<&'static str>) -> Result<Option<Keyed<Key>>, Error> {
    match key::read(row, also) {
        Ok(key) => Ok(Some(key)),
        Err(column) => declined(column),
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

/// What the core reads of this host beside `general_settings`. The gateway guards its inference
/// routes alone, so it names no route group, and it serves no UI login. The proxy budget, the
/// license and the environment only shape steps and text this host does not answer with.
fn settings(config: &Config) -> Settings {
    Settings::new(
        &config.general_settings,
        HostFacts {
            master_key: master_key(config).map(|key| MasterKey::new(key.clone())),
            premium: false,
            has_db: true,
            db_configured: true,
            ui_login: false,
            proxy_budget: false,
            metrics_require_auth: true,
            route_groups: RouteGroups::default(),
        },
        &|_| None,
    )
}

/// Verifies the master key of `config` and the virtual keys of `rows`, read from `cache` first
/// when the gateway shares Python's.
pub fn authenticator(
    config: &Config,
    rows: Arc<dyn KeyRows>,
    cache: Option<AuthCache>,
) -> Arc<dyn Authenticator> {
    let host = NativeAuthHost {
        rows,
        cache,
        master: master_key(config).map(|key| Token::new(key.clone()).hash()),
    };
    Arc::new(KeyAuthenticator::new(host, settings(config)))
}

struct NoOperatorCode;

impl OperatorHooks for NoOperatorCode {
    async fn custom_validate(&self, _: &Value) -> Result<bool, Error> {
        declined("custom_validate")
    }
}

impl AuthHost for NativeAuthHost {
    type Rows = NativeRows;

    /// Read from Redis alone, since this process hears no evictions and a copy in its memory
    /// could outlive a block.
    async fn cached_key(&self, hash: &KeyHash) -> Result<Option<Keyed<Key>>, Error> {
        if self.master.as_ref() == Some(hash) {
            return Ok(None);
        }
        let Some(cache) = &self.cache else {
            return Ok(None);
        };
        match cached::read(cache, hash.as_str()).await {
            Some(cached) => keyed(&cached.row, cached.carried),
            None => Ok(None),
        }
    }

    /// A key the native app answers is written back to the cache Python fills, under a lease
    /// taken before the read. Any other key reaches Python, which caches it itself.
    async fn db_key(&self, hash: &KeyHash, _: &()) -> Result<Option<Keyed<Key>>, Error> {
        let lease = match &self.cache {
            Some(cache) => cached::lease(cache, hash.as_str()).await,
            None => None,
        };
        let found = self.rows.find(hash.as_str()).await.map_err(|error| {
            tracing::warn!(
                error = %litellm_error::Report::new(&error),
                "a virtual key was not read from the database"
            );
            Error::HostFailed
        })?;
        let Some(row) = found else {
            return Ok(None);
        };
        let key = keyed(&row, None)?;
        if let (Some(cache), Some(key)) = (&self.cache, &key)
            && key::is_answered_natively(key)
        {
            cached::write(cache, hash.as_str(), lease.as_deref(), &row).await;
        }
        Ok(key)
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
pub(crate) mod tests {
    use std::{
        collections::{BTreeMap, BTreeSet},
        sync::atomic::{AtomicUsize, Ordering},
        time::SystemTime,
    };

    use chrono::{DateTime, NaiveDateTime};
    use litellm_gateway_auth::{
        Auth, AuthenticatedRequest, AuthenticationMethod, Error as Rejected, Permissions,
        hash_token,
        keys::{KeyLimits, ModelLimits},
    };
    use litellm_redis_testing::RedisServer;
    use litellm_repositories::RawJson;
    use litellm_secret_value::SecretValue;
    use rstest::rstest;
    use serde_json::value::RawValue;

    use super::*;

    const MASTER_KEY: &str = "sk-master-credential";
    const VIRTUAL_KEY: &str = "sk-test-credential";

    fn json(text: &str) -> RawJson {
        RawJson(RawValue::from_string(text.to_owned()).unwrap())
    }

    fn at(second: i64) -> NaiveDateTime {
        DateTime::from_timestamp(second, 0).unwrap().naive_utc()
    }

    fn far_future() -> NaiveDateTime {
        at(4_102_444_800)
    }

    /// The row of a key generated with no parameters, as `/key/generate` stores it, under the
    /// hash of `token`.
    pub(crate) fn plain(token: &str) -> KeyView {
        let row: KeyView =
            serde_json::from_str(include_str!("../../../tests/keys/plain.json")).unwrap();
        KeyView {
            token: hash_token(token),
            ..row
        }
    }

    /// The rows of a database that counts each read.
    #[derive(Default)]
    struct Counted {
        rows: BTreeMap<String, KeyView>,
        reads: AtomicUsize,
    }

    impl KeyRows for Counted {
        fn find<'a>(&'a self, hash: &'a str) -> RowFuture<'a> {
            self.reads.fetch_add(1, Ordering::SeqCst);
            self.rows.find(hash)
        }
    }

    struct Unreachable;

    impl KeyRows for Unreachable {
        fn find<'a>(&'a self, _: &'a str) -> RowFuture<'a> {
            Box::pin(async { Err(Source::new(std::io::Error::other("database down"))) })
        }
    }

    fn config() -> Config {
        Config::from_yaml(&format!("general_settings:\n  master_key: {MASTER_KEY}\n")).unwrap()
    }

    async fn authenticate(
        rows: Arc<dyn KeyRows>,
        token: &str,
    ) -> Result<AuthenticatedRequest, Rejected> {
        Auth::with_keys(&config(), authenticator(&config(), rows, None))
            .authenticate(&SecretValue::new(token))
            .await
    }

    /// The database holding `key` alone, under the hash of [`VIRTUAL_KEY`].
    fn holding(change: fn(&mut KeyView)) -> Arc<Counted> {
        let mut key = plain(VIRTUAL_KEY);
        change(&mut key);
        Arc::new(Counted {
            rows: BTreeMap::from([(key.token.clone(), key)]),
            reads: AtomicUsize::new(0),
        })
    }

    #[rstest]
    #[case::as_generated(|_: &mut KeyView| {}, None, &[])]
    #[case::no_key_type(|key: &mut KeyView| key.key_type = None, None, &[])]
    #[case::unblocked(|key: &mut KeyView| key.blocked = Some(false), None, &[])]
    #[case::expiring(|key: &mut KeyView| key.expires = Some(far_future()), Some(far_future()), &[])]
    #[case::with_a_model_list(
        |key: &mut KeyView| key.models = vec!["claude".to_owned(), "haiku".to_owned()],
        None,
        &["claude", "haiku"]
    )]
    #[case::reaching_every_model(
        |key: &mut KeyView| key.models = vec!["*".to_owned(), "all-proxy-models".to_owned()],
        None,
        &["*", "all-proxy-models"]
    )]
    #[case::empty_json_columns(
        |key: &mut KeyView| {
            key.metadata = json("[]");
            key.budget_limits = Some(json("null"));
            key.aliases = json("{}");
        },
        None,
        &[]
    )]
    #[tokio::test]
    async fn a_key_that_holds_only_what_the_native_app_honors_is_authenticated(
        #[case] change: fn(&mut KeyView),
        #[case] expires_at: Option<NaiveDateTime>,
        #[case] models: &[&str],
    ) {
        let identity = authenticate(holding(change), VIRTUAL_KEY).await.unwrap();

        let caller = identity.caller();
        let listed: Vec<&str> = match caller.restrictions() {
            Permissions::Models(models) => models.iter().map(String::as_str).collect(),
            _ => Vec::new(),
        };
        assert_eq!(
            (
                &caller.authentication().method,
                caller.authentication().credential_id.as_str(),
                caller.authentication().expires_at,
                listed,
            ),
            (
                &AuthenticationMethod::VirtualKey,
                hash_token(VIRTUAL_KEY).as_str(),
                expires_at.map(|at| SystemTime::from(at.and_utc())),
                models.to_vec(),
            )
        );
    }

    #[rstest]
    #[case::key_type(|key: &mut KeyView| key.key_type = Some("management".to_owned()))]
    #[case::soft_budget_cooldown(|key: &mut KeyView| key.soft_budget_cooldown = true)]
    #[case::a_model_pattern(|key: &mut KeyView| key.models = vec!["claude".to_owned(), "anthropic/*".to_owned()])]
    #[case::the_team_models(|key: &mut KeyView| key.models = vec!["all-team-models".to_owned()])]
    #[case::no_default_models(|key: &mut KeyView| key.models = vec!["no-default-models".to_owned()])]
    #[case::aliases(|key: &mut KeyView| key.aliases = json(r#"{"a":"b"}"#))]
    #[case::config(|key: &mut KeyView| key.config = json(r#"{"a":1}"#))]
    #[case::router_settings(|key: &mut KeyView| key.router_settings = Some(json(r#"{"a":1}"#)))]
    #[case::user_id(|key: &mut KeyView| key.user_id = Some("user".to_owned()))]
    #[case::an_empty_user_id(|key: &mut KeyView| key.user_id = Some(String::new()))]
    #[case::team_id(|key: &mut KeyView| key.team_id = Some("team".to_owned()))]
    #[case::the_ui_team(|key: &mut KeyView| key.team_id = Some("litellm-dashboard".to_owned()))]
    #[case::agent_id(|key: &mut KeyView| key.agent_id = Some("agent".to_owned()))]
    #[case::project_id(|key: &mut KeyView| key.project_id = Some("project".to_owned()))]
    #[case::org_id(|key: &mut KeyView| key.org_id = Some("org".to_owned()))]
    #[case::permissions(|key: &mut KeyView| key.permissions = json(r#"{"a":true}"#))]
    #[case::metadata(|key: &mut KeyView| key.metadata = json(r#"{"guardrails":["pii"]}"#))]
    #[case::metadata_beside_a_model_limit(|key: &mut KeyView| key.metadata = json(r#"{"model_rpm_limit":{"claude":2},"tags":["a"]}"#))]
    #[case::a_model_limit_that_is_no_whole_number(|key: &mut KeyView| key.metadata = json(r#"{"model_tpm_limit":{"claude":"100"}}"#))]
    #[case::model_limits_that_are_no_map(|key: &mut KeyView| key.metadata = json(r#"{"model_rpm_limit":5}"#))]
    #[case::max_budget(|key: &mut KeyView| key.max_budget = Some(5.0))]
    #[case::soft_budget(|key: &mut KeyView| key.soft_budget = Some(5.0))]
    #[case::budget_duration(|key: &mut KeyView| key.budget_duration = Some("30d".to_owned()))]
    #[case::budget_reset_at(|key: &mut KeyView| key.budget_reset_at = Some(at(9)))]
    #[case::budget_limits(|key: &mut KeyView| key.budget_limits = Some(json(r#"[{"a":1}]"#)))]
    #[case::budget_fallbacks(|key: &mut KeyView| key.budget_fallbacks = json(r#"{"a":1}"#))]
    #[case::budget_id(|key: &mut KeyView| key.budget_id = Some("budget".to_owned()))]
    #[case::model_max_budget(|key: &mut KeyView| key.model_max_budget = json(r#"{"claude":1}"#))]
    #[case::allowed_cache_controls(|key: &mut KeyView| key.allowed_cache_controls = Some(vec!["no-cache".to_owned()]))]
    #[case::allowed_routes(|key: &mut KeyView| key.allowed_routes = Some(vec!["/v1/messages".to_owned()]))]
    #[case::access_group_ids(|key: &mut KeyView| key.access_group_ids = Some(vec!["group".to_owned()]))]
    #[case::object_permission_id(|key: &mut KeyView| key.object_permission_id = Some("permission".to_owned()))]
    #[tokio::test]
    async fn a_key_that_holds_anything_else_is_not_native(#[case] change: fn(&mut KeyView)) {
        let refused = authenticate(holding(change), VIRTUAL_KEY).await;

        assert!(
            matches!(refused, Err(Rejected::NotNative)),
            "{:?}",
            refused.err()
        );
    }

    #[rstest]
    #[case::rpm(|key: &mut KeyView| key.rpm_limit = Some(2), KeyLimits { rpm: Some(2), ..KeyLimits::default() })]
    #[case::tpm(|key: &mut KeyView| key.tpm_limit = Some(100), KeyLimits { tpm: Some(100), ..KeyLimits::default() })]
    #[case::parallel(
        |key: &mut KeyView| key.max_parallel_requests = Some(1),
        KeyLimits { max_parallel_requests: Some(1), ..KeyLimits::default() }
    )]
    #[case::a_limit_of_zero(|key: &mut KeyView| key.rpm_limit = Some(0), KeyLimits { rpm: Some(0), ..KeyLimits::default() })]
    #[case::per_model_in_its_metadata(
        |key: &mut KeyView| key.metadata = json(r#"{"model_rpm_limit":{"claude":2},"model_tpm_limit":{"claude":100,"gpt":0}}"#),
        KeyLimits {
            model_rpm: ModelLimits::new(BTreeMap::from([("claude".to_owned(), 2)])),
            model_tpm: ModelLimits::new(BTreeMap::from([("claude".to_owned(), 100), ("gpt".to_owned(), 0)])),
            ..KeyLimits::default()
        }
    )]
    #[case::a_daily_token_limit_that_only_counts_batches(|key: &mut KeyView| key.tpd_limit = Some(1), KeyLimits::default())]
    #[tokio::test]
    async fn a_keys_limits_reach_its_caller(
        #[case] change: fn(&mut KeyView),
        #[case] limits: KeyLimits,
    ) {
        let identity = authenticate(holding(change), VIRTUAL_KEY).await.unwrap();

        assert_eq!(identity.caller().authentication().limits, limits);
    }

    #[rstest]
    #[case::alone(|key: &mut KeyView| key.blocked = Some(true))]
    #[case::of_a_team(|key: &mut KeyView| {
        key.blocked = Some(true);
        key.team_id = Some("team".to_owned());
    })]
    #[case::unknown(|key: &mut KeyView| key.token = "another".to_owned())]
    #[tokio::test]
    async fn a_blocked_or_unknown_key_is_an_invalid_token(#[case] change: fn(&mut KeyView)) {
        let refused = authenticate(holding(change), VIRTUAL_KEY).await;

        assert!(
            matches!(refused, Err(Rejected::InvalidToken)),
            "{:?}",
            refused.err()
        );
    }

    #[tokio::test]
    async fn an_expired_key_is_expired() {
        let refused = authenticate(
            holding(|key: &mut KeyView| key.expires = Some(at(1))),
            VIRTUAL_KEY,
        )
        .await;

        assert!(
            matches!(refused, Err(Rejected::Expired)),
            "{:?}",
            refused.err()
        );
    }

    #[rstest]
    #[case::the_master_key(MASTER_KEY, Some(AuthenticationMethod::MasterKey))]
    #[case::a_token_that_is_not_a_key("eyJhbGciOiJSUzI1NiJ9.e30.c2ln", None)]
    #[tokio::test]
    async fn only_a_virtual_key_is_read_from_the_database(
        #[case] token: &str,
        #[case] method: Option<AuthenticationMethod>,
    ) {
        let rows = holding(|_| {});

        let authenticated = authenticate(rows.clone(), token).await;

        assert_eq!(
            authenticated
                .ok()
                .map(|identity| identity.caller().authentication().method.clone()),
            method
        );
        assert_eq!(rows.reads.load(Ordering::SeqCst), 0);
    }

    /// The auth cache `AppState` builds over `redis` with `enable_redis_auth_cache` on.
    async fn cache_on(redis: &RedisServer) -> AuthCache {
        let config = Config::from_yaml(&format!(
            "general_settings:\n  coordination_redis:\n    host: 127.0.0.1\n    port: {}\nlitellm_settings:\n  enable_redis_auth_cache: true\n",
            redis.port()
        ))
        .unwrap();
        let resources = crate::Resources::build(&config).unwrap();
        let clients = crate::redis::connect(&config, &resources).await.unwrap();
        crate::state::auth_cache(&config, &clients).unwrap()
    }

    async fn authenticate_behind(
        cache: AuthCache,
        rows: Arc<dyn KeyRows>,
    ) -> Result<AuthenticatedRequest, Rejected> {
        Auth::with_keys(&config(), authenticator(&config(), rows, Some(cache)))
            .authenticate(&SecretValue::new(VIRTUAL_KEY))
            .await
    }

    /// Whether `redis` holds an entry for [`VIRTUAL_KEY`].
    async fn is_cached(redis: &RedisServer) -> bool {
        let held = redis.cli(&["EXISTS", &hash_token(VIRTUAL_KEY)]).await;
        held.unwrap().trim() == "1"
    }

    #[tokio::test]
    async fn a_key_read_from_the_database_is_cached_as_python_caches_it() {
        let redis = RedisServer::start().await.unwrap();
        let rows = holding(|_| {});
        let auth = Auth::with_keys(
            &config(),
            authenticator(&config(), rows.clone(), Some(cache_on(&redis).await)),
        );
        let hash = hash_token(VIRTUAL_KEY);

        let first = auth.authenticate(&SecretValue::new(VIRTUAL_KEY)).await;
        let second = auth.authenticate(&SecretValue::new(VIRTUAL_KEY)).await;

        let cached: serde_json::Value =
            serde_json::from_str(&redis.cli(&["GET", &hash]).await.unwrap()).unwrap();
        let python: serde_json::Value =
            serde_json::from_str(include_str!("../../../tests/keys/plain.json")).unwrap();
        let fields = |entry: &serde_json::Value| -> BTreeSet<String> {
            entry.as_object().unwrap().keys().cloned().collect()
        };
        let written = fields(&cached);
        let expected: BTreeSet<String> = fields(&python)
            .into_iter()
            .chain(["last_refreshed_at".to_owned()])
            .collect();
        let ttl: i64 = redis
            .cli(&["TTL", &hash])
            .await
            .unwrap()
            .trim()
            .parse()
            .unwrap();
        assert_eq!(
            (
                first.is_ok(),
                second.is_ok(),
                rows.reads.load(Ordering::SeqCst),
                written,
                cached["last_refreshed_at"].is_f64(),
                (1..=60).contains(&ttl),
            ),
            (true, true, 1, expected, true, true)
        );
    }

    /// A database whose read of the key returns after another pod evicted it, as a block that
    /// lands mid-read does.
    struct EvictedMidRead {
        rows: BTreeMap<String, KeyView>,
        another_pod: AuthCache,
    }

    impl KeyRows for EvictedMidRead {
        fn find<'a>(&'a self, hash: &'a str) -> RowFuture<'a> {
            Box::pin(async move {
                self.another_pod.delete(&[hash.to_owned()]).await.unwrap();
                self.rows.find(hash).await
            })
        }
    }

    #[tokio::test]
    async fn a_key_evicted_while_it_was_read_is_not_cached() {
        let redis = RedisServer::start().await.unwrap();
        let key = plain(VIRTUAL_KEY);
        let rows = Arc::new(EvictedMidRead {
            rows: BTreeMap::from([(key.token.clone(), key)]),
            another_pod: cache_on(&redis).await,
        });

        let authenticated = authenticate_behind(cache_on(&redis).await, rows).await;

        assert_eq!(
            (authenticated.is_ok(), is_cached(&redis).await),
            (true, false)
        );
    }

    #[rstest]
    #[case::with_an_owner(|key: &mut KeyView| key.user_id = Some("user".to_owned()))]
    #[case::in_a_team(|key: &mut KeyView| key.team_id = Some("team".to_owned()))]
    #[case::in_a_project(|key: &mut KeyView| key.project_id = Some("project".to_owned()))]
    #[case::with_a_budget(|key: &mut KeyView| key.max_budget = Some(5.0))]
    #[case::with_aliases(|key: &mut KeyView| key.aliases = json(r#"{"a":"b"}"#))]
    #[case::blocked(|key: &mut KeyView| key.blocked = Some(true))]
    #[tokio::test]
    async fn a_key_python_answers_is_left_for_python_to_cache(#[case] change: fn(&mut KeyView)) {
        let redis = RedisServer::start().await.unwrap();

        let authenticated = authenticate_behind(cache_on(&redis).await, holding(change)).await;

        assert_eq!(
            (authenticated.is_err(), is_cached(&redis).await),
            (true, false)
        );
    }

    #[tokio::test]
    async fn a_database_that_fails_is_unavailable_and_never_a_denial() {
        let Err(refused) = authenticate(Arc::new(Unreachable), VIRTUAL_KEY).await else {
            panic!("a key was authenticated with no database to read it from");
        };

        assert!(matches!(refused, Rejected::Unavailable(_)), "{refused:?}");
        assert!(!refused.is_denied());
    }
}
