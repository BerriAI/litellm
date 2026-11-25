//! A virtual key, the team, project, membership and user a caller belongs to, and the end user a
//! request names, read through the auth cache like the proxy's `get_key_object` and its siblings:
//! the cache, then the database on a miss, which caches the row in the shape every pod reads, for
//! every pod when the cache has the shared Redis attached. Nothing guards that write against an
//! eviction that lands during the database read.

use std::{
    future::Future,
    sync::Arc,
    time::{Duration, SystemTime},
};

use litellm_auth_store::{
    AliasMatch, EndUserDatabase, Error, KeyDatabase, OrganizationDatabase, ProjectDatabase,
    TeamDatabase, TeamMembershipDatabase, UserDatabase,
};
use litellm_gateway_auth_core::{KeyHash, authn::jwt::MappedClaim};
use litellm_redis_store::{
    FromValue, IntoValue,
    auth_cache::{
        AuthCache, AuthObject, EndUserEntry, KeyEntry, OrganizationEntry, ProjectEntry,
        RestrictedEndUsers, Settings, TeamEntry, TeamMembershipEntry, TeamMembershipReservation,
        UserEntry,
    },
};
use litellm_repositories::{
    Database, EndUserRepository, JwtKeyMappingRepository, KeyRepository,
    ObjectPermissionRepository, OrganizationRepository, ProjectRepository, Side,
    TeamMembershipRepository, TeamRepository, UserRepository,
};
use serde_json::Value;
use tokio::sync::Mutex;

const ORG_CACHE_TTL: Duration = Duration::from_secs(5);
/// `END_USER_RESTRICTED_REGISTRY_MAX_SIZE`.
const RESTRICTED_IDS_MAX: usize = 5000;
/// `REGISTRY_ERROR_NEGATIVE_CACHE_TTL`.
const RESTRICTED_IDS_ERROR_TTL: Duration = Duration::from_secs(30);
/// How long `bounded_db_lookup` gives an auth read, the default of
/// `PROXY_DB_LOOKUP_DEADLINE_SECONDS`, a variable the gateway does not read yet.
const LOOKUP_DEADLINE: Duration = Duration::from_secs(10);

pub struct AuthStore {
    cache: Arc<AuthCache>,
    keys: KeyDatabase,
    teams: TeamDatabase,
    organizations: OrganizationDatabase,
    projects: ProjectDatabase,
    memberships: TeamMembershipDatabase,
    users: UserDatabase,
    mappings: JwtKeyMappingRepository,
    end_users: EndUserDatabase,
    /// Lets one request at a time load the restricted end-user ids (`_END_USER_REGISTRY_LOAD_LOCK`).
    restricted_ids_loading: Mutex<()>,
    lookup_deadline: Duration,
}

/// What the store read of an end user.
pub enum EndUserRead {
    Row(Box<EndUserEntry>),
    /// No row, or a row that restricts nothing.
    NoRow,
    /// The database did not give the row.
    Unread,
}

impl AuthStore {
    /// The store over `database`, with a repository for each object it reads.
    pub fn new(cache: Arc<AuthCache>, database: Database) -> Self {
        let object_permissions = Arc::new(ObjectPermissionRepository::new(database.clone()));
        Self {
            cache,
            keys: KeyDatabase::new(
                Arc::new(KeyRepository::new(database.clone())),
                object_permissions.clone(),
            ),
            teams: TeamDatabase::new(
                Arc::new(TeamRepository::new(database.clone())),
                object_permissions,
            ),
            projects: ProjectDatabase::new(Arc::new(ProjectRepository::new(database.clone()))),
            organizations: OrganizationDatabase::new(Arc::new(OrganizationRepository::new(
                database.clone(),
            ))),
            memberships: TeamMembershipDatabase::new(Arc::new(TeamMembershipRepository::new(
                database.clone(),
            ))),
            users: UserDatabase::new(Arc::new(UserRepository::new(database.clone()))),
            mappings: JwtKeyMappingRepository::new(database.clone()),
            end_users: EndUserDatabase::new(Arc::new(EndUserRepository::new(database))),
            restricted_ids_loading: Mutex::new(()),
            lookup_deadline: LOOKUP_DEADLINE,
        }
    }

    #[cfg(test)]
    fn with_lookup_deadline(self, lookup_deadline: Duration) -> Self {
        Self {
            lookup_deadline,
            ..self
        }
    }

    /// The key the cache holds for `hash`.
    pub async fn cached(&self, hash: &KeyHash) -> Option<KeyEntry> {
        self.read(&object(hash)).await
    }

    /// A cache that fails and an entry that does not decode are both a miss, as they are to
    /// `UserApiKeyCache`.
    async fn read<T: FromValue>(&self, object: &AuthObject) -> Option<T> {
        let cached = self
            .cache
            .get(None, object)
            .await
            .inspect_err(|error| {
                tracing::warn!(
                    error = %litellm_error::Report::new(error),
                    key = object.key(),
                    "an object was not read from the auth cache"
                );
            })
            .ok()
            .flatten()?;
        T::from_value(&cached)
            .inspect_err(|error| {
                tracing::warn!(
                    error = %litellm_error::Report::new(error),
                    key = object.key(),
                    "a cached object does not decode and is read as a miss"
                );
            })
            .ok()
    }

    /// Cached for `ttl`, or for the cache's own TTL, as `get_management_object_ttl` names
    /// `default_in_memory_ttl`.
    async fn write(&self, object: &AuthObject, value: impl IntoValue, ttl: Option<Duration>) {
        if let Err(error) = self.cache.set(None, object, value.into_value(), ttl).await {
            tracing::warn!(
                error = %litellm_error::Report::new(&error),
                key = object.key(),
                "an object was not cached"
            );
        }
    }

    /// The key the database holds for `hash`, cached. A key the cache did not take is returned
    /// all the same.
    pub async fn loaded(&self, hash: &KeyHash) -> Result<Option<KeyEntry>, Error> {
        let Some(key) = Box::pin(self.keys.read(hash, Side::Read)).await? else {
            return Ok(None);
        };
        self.write(&object(hash), key.clone(), None).await;
        Ok(Some(key))
    }

    /// `get_team_object`: the cached team, else the database's, cached with `last_refreshed_at`
    /// set to the read time, like `cache_team_object`.
    pub async fn team(&self, team_id: &str) -> Result<Option<TeamEntry>, Error> {
        let object = AuthObject::Team(team_id.to_owned());
        if let Some(team) = self.read(&object).await {
            return Ok(Some(team));
        }
        let Some(team) = self.teams.team(team_id, Side::Read).await? else {
            return Ok(None);
        };
        let team = TeamEntry {
            last_refreshed_at: Some(now()),
            ..team
        };
        self.write(&object, team.clone(), None).await;
        Ok(Some(team))
    }

    /// `get_project_object`: the cached project, else the database's, cached with
    /// `last_refreshed_at` set to the read time, like `get_project_object`.
    pub async fn project(&self, project_id: &str) -> Result<Option<ProjectEntry>, Error> {
        let object = AuthObject::Project(project_id.to_owned());
        if let Some(project) = self.read(&object).await {
            return Ok(Some(project));
        }
        let Some(project) = self.projects.project(project_id).await? else {
            return Ok(None);
        };
        let project = ProjectEntry {
            last_refreshed_at: Some(now()),
            ..project
        };
        self.write(&object, project.clone(), None).await;
        Ok(Some(project))
    }

    /// `get_team_object_by_alias`: the team cached under the alias, else the one row the database
    /// holds with it, cached under the alias and the team's id.
    pub async fn team_by_alias(&self, alias: &str) -> Result<AliasMatch<TeamEntry>, Error> {
        let object = AuthObject::TeamAlias(alias.to_owned());
        if let Some(team) = self.read(&object).await {
            return Ok(AliasMatch::Found(team));
        }
        let found = self.teams.team_by_alias(alias).await?;
        let AliasMatch::Found(team) = &found else {
            return Ok(found);
        };
        self.write(&object, team.clone(), None).await;
        self.write(&AuthObject::Team(team.team_id.clone()), team.clone(), None)
            .await;
        Ok(found)
    }

    /// The budget-inclusive organization from the cache or the database, cached only when found.
    pub async fn org(&self, organization_id: &str) -> Result<Option<OrganizationEntry>, Error> {
        let object = AuthObject::OrgWithBudget(organization_id.to_owned());
        if let Some(org) = self.read(&object).await {
            return Ok(Some(org));
        }
        let read = self.organizations.organization(organization_id, true);
        let org =
            litellm_db::within(Some(self.lookup_deadline), async { Ok(read.await) }).await??;
        if let Some(org) = &org {
            self.write(&object, org.clone(), Some(ORG_CACHE_TTL)).await;
        }
        Ok(org)
    }

    /// `get_team_membership`: the cached membership, or the cached word that there is none, else
    /// the database's answer, cached either way.
    pub async fn team_membership(
        &self,
        user_id: &str,
        team_id: &str,
    ) -> Result<Option<TeamMembershipEntry>, Error> {
        let object = AuthObject::TeamMembershipReservation {
            user_id: user_id.to_owned(),
            team_id: team_id.to_owned(),
        };
        match self.read(&object).await {
            Some(TeamMembershipReservation::Member(membership)) => return Ok(Some(membership)),
            Some(TeamMembershipReservation::NotMember) => return Ok(None),
            None => {}
        }
        let membership = self
            .memberships
            .team_membership(user_id, team_id, Side::Read)
            .await?;
        let cached = membership.clone().map_or(
            TeamMembershipReservation::NotMember,
            TeamMembershipReservation::Member,
        );
        self.write(&object, cached, None).await;
        Ok(membership)
    }

    /// `get_user_object` for a user read by its id: the cached user, else the database's, cached.
    pub async fn user(&self, user_id: &str) -> Result<Option<UserEntry>, Error> {
        let object = AuthObject::User(user_id.to_owned());
        if let Some(user) = self.read(&object).await {
            return Ok(Some(user));
        }
        let Some(user) = self.users.user(user_id, Side::Read).await? else {
            return Ok(None);
        };
        self.write(&object, user.clone(), None).await;
        Ok(Some(user))
    }

    /// `get_end_user_object`: the cached end user, else no row for an id the restricted ids leave
    /// out, else the database's row, cached.
    ///
    /// An id the restricted ids leave out has no row or a row that restricts nothing, so a request
    /// that names such an end user costs no database read. A missing row is not cached, so a row
    /// created later is read on that end user's next request.
    pub async fn end_user(&self, end_user_id: &str) -> EndUserRead {
        let object = AuthObject::EndUser(end_user_id.to_owned());
        if let Some(end_user) = self.read::<EndUserEntry>(&object).await {
            return EndUserRead::Row(Box::new(end_user));
        }
        let restricted = self.restricted_ids().await;
        if restricted.is_some_and(|ids| !ids.iter().any(|id| id == end_user_id)) {
            return EndUserRead::NoRow;
        }
        let row = self.end_users.end_user(end_user_id);
        match self.bounded("end_user", row).await {
            Some(Some(end_user)) => {
                self.write(&object, end_user.clone(), None).await;
                EndUserRead::Row(Box::new(end_user))
            }
            Some(None) => EndUserRead::NoRow,
            None => EndUserRead::Unread,
        }
    }

    /// `_load_end_user_restricted_registry`: the ids of the end users whose row restricts
    /// something, cached whole and loaded by one request at a time. `None` when there are more
    /// than the cache holds or the database did not list them in time, which the cache remembers
    /// too, so every end user is read by its id until that expires.
    async fn restricted_ids(&self) -> Option<Vec<String>> {
        let object = AuthObject::EndUserRestrictedRegistry;
        let cached = async || self.read::<RestrictedEndUsers>(&object).await.map(usable);
        if let Some(ids) = cached().await {
            return ids;
        }
        let _loading = match self.restricted_ids_loading.try_lock() {
            Ok(loading) => loading,
            Err(_) => {
                let loading = self.restricted_ids_loading.lock().await;
                if let Some(ids) = cached().await {
                    return ids;
                }
                loading
            }
        };
        let limit = i64::try_from(RESTRICTED_IDS_MAX + 1).unwrap_or(i64::MAX);
        let listing = self.end_users.restricted_ids(limit);
        match self.bounded("end_user_registry", listing).await {
            Some(ids) if ids.len() > RESTRICTED_IDS_MAX => {
                self.write(&object, RestrictedEndUsers::Overflow, None)
                    .await;
                None
            }
            Some(ids) => {
                self.write(&object, RestrictedEndUsers::Ids(ids.clone()), None)
                    .await;
                Some(ids)
            }
            None => {
                let ttl = Some(RESTRICTED_IDS_ERROR_TTL);
                self.write(&object, RestrictedEndUsers::Overflow, ttl).await;
                None
            }
        }
    }

    /// `bounded_db_lookup`: `read`, or `None` when it fails or does not answer within the
    /// deadline, so a database that stalls holds no request, and no lock, longer than that.
    async fn bounded<T>(
        &self,
        lookup: &'static str,
        read: impl Future<Output = Result<T, Error>>,
    ) -> Option<T> {
        match tokio::time::timeout(self.lookup_deadline, read).await {
            Ok(Ok(read)) => Some(read),
            Ok(Err(error)) => {
                tracing::warn!(
                    error = %litellm_error::Report::new(&error),
                    lookup,
                    "an auth object was not read from the database"
                );
                None
            }
            Err(_) => {
                tracing::warn!(
                    lookup,
                    deadline = ?self.lookup_deadline,
                    "the database did not answer an auth read in time"
                );
                None
            }
        }
    }

    /// `_resize_the_auth_cache` on a reload: `entries` in each partition from the next write on,
    /// with the TTL startup set kept.
    pub(crate) fn resize(&self, entries: usize) {
        self.cache.reconfigure(Settings {
            memory_entries: entries,
            ..self.cache.settings()
        });
    }

    pub async fn cached_key_mapping(&self, claim: MappedClaim<'_>) -> Option<Value> {
        self.read::<String>(&mapping_object(claim))
            .await
            .map(Value::String)
    }

    pub async fn cache_key_mapping(&self, claim: MappedClaim<'_>, value: &Value, ttl: Duration) {
        if let Some(text) = value.as_str() {
            self.write(&mapping_object(claim), text.to_owned(), Some(ttl))
                .await;
        }
    }

    pub async fn key_mapping_hash(
        &self,
        claim_field: &str,
        claim_value: &str,
        issuer: Option<&str>,
    ) -> Result<Option<KeyHash>, litellm_db::Error> {
        self.mappings
            .mapped_token(claim_field, claim_value, issuer)
            .await
            .map(|token| token.map(KeyHash::new))
    }
}

/// The cached ids, or `None` when they overflowed and every end user is read by its id.
fn usable(cached: RestrictedEndUsers) -> Option<Vec<String>> {
    match cached {
        RestrictedEndUsers::Ids(ids) => Some(ids),
        RestrictedEndUsers::Overflow => None,
    }
}

/// `time.time()`, which `cache_team_object` stamps a team with.
fn now() -> f64 {
    SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs_f64()
}

fn object(hash: &KeyHash) -> AuthObject {
    AuthObject::Key(hash.as_str().to_owned())
}

fn mapping_object(claim: MappedClaim<'_>) -> AuthObject {
    AuthObject::JwtKeyMapping {
        issuer: claim.issuer.map(str::to_owned),
        claim_name: claim.field.to_owned(),
        claim_value: claim.value.to_owned(),
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use serde_json::json;

    use super::*;
    use crate::host::auth::keys::{self, Held};

    const ROWS: &str = r#"INSERT INTO "LiteLLM_EndUserTable" (user_id, spend, blocked) VALUES
        ('eu-spent', 3, false), ('eu-spent-too', 4, false), ('eu-blocked', 0, true)"#;
    const A_LATER_BLOCK: &str = r#"INSERT INTO "LiteLLM_EndUserTable" (user_id, spend, blocked)
        VALUES ('eu-blocked-later', 0, true)"#;
    const AS_MANY_BLOCKED_AS_THE_CACHE_HOLDS: &str = r#"INSERT INTO "LiteLLM_EndUserTable"
        (user_id, spend, blocked) SELECT 'eu-many-' || n, 0, true FROM generate_series(1, 4999) n"#;
    const ONE_MORE_BLOCKED: &str = r#"INSERT INTO "LiteLLM_EndUserTable" (user_id, spend, blocked)
        VALUES ('eu-one-more', 0, true)"#;
    const NO_BLOCKED: &str = r#"DELETE FROM "LiteLLM_EndUserTable" WHERE blocked"#;
    const NO_ROWS: &str = r#"DELETE FROM "LiteLLM_EndUserTable""#;

    async fn stored() -> (AuthStore, Held) {
        let held = Held::start().await;
        held.execute(ROWS).await;
        let store = AuthStore::new(keys::memory_cache(), held.database())
            .with_lookup_deadline(Duration::from_millis(500));
        (store, held)
    }

    /// The spend of the row `store` reads for `end_user_id`, `None` when it reads no row, and
    /// an error when the database did not give it.
    async fn spend(store: &AuthStore, end_user_id: &str) -> Result<Option<f64>, ()> {
        match store.end_user(end_user_id).await {
            EndUserRead::Row(end_user) => Ok(Some(end_user.spend)),
            EndUserRead::NoRow => Ok(None),
            EndUserRead::Unread => Err(()),
        }
    }

    #[rstest]
    #[case::a_row_that_restricts_something("eu-blocked", Ok(Some(0.0)))]
    #[case::a_row_that_only_describes_the_end_user("eu-spent", Ok(None))]
    #[case::no_row("eu-new", Ok(None))]
    #[tokio::test(flavor = "multi_thread")]
    async fn only_an_end_user_among_the_restricted_ids_has_its_row_read(
        #[case] end_user_id: &str,
        #[case] read: Result<Option<f64>, ()>,
    ) {
        let (store, _held) = stored().await;

        assert_eq!(spend(&store, end_user_id).await, read);
    }

    #[rstest]
    #[tokio::test(flavor = "multi_thread")]
    async fn a_row_and_the_restricted_ids_are_cached_once_read() {
        let (store, held) = stored().await;

        let from_database = spend(&store, "eu-blocked").await;
        held.execute(NO_ROWS).await;
        held.execute(A_LATER_BLOCK).await;

        assert_eq!(
            (
                from_database,
                spend(&store, "eu-blocked").await,
                spend(&store, "eu-blocked-later").await
            ),
            (Ok(Some(0.0)), Ok(Some(0.0)), Ok(None))
        );
    }

    #[rstest]
    #[tokio::test(flavor = "multi_thread")]
    async fn as_many_restricted_ids_as_the_cache_holds_still_spare_the_others_a_read() {
        let (store, held) = stored().await;
        held.execute(AS_MANY_BLOCKED_AS_THE_CACHE_HOLDS).await;

        assert_eq!(spend(&store, "eu-spent").await, Ok(None));
    }

    #[rstest]
    #[tokio::test(flavor = "multi_thread")]
    async fn more_restricted_ids_than_the_cache_holds_read_every_end_user_by_its_id() {
        let (store, held) = stored().await;
        held.execute(AS_MANY_BLOCKED_AS_THE_CACHE_HOLDS).await;
        held.execute(ONE_MORE_BLOCKED).await;

        let by_id = spend(&store, "eu-spent").await;
        held.execute(NO_BLOCKED).await;

        assert_eq!(
            (
                by_id,
                spend(&store, "eu-spent-too").await,
                spend(&store, "eu-new").await
            ),
            (Ok(Some(3.0)), Ok(Some(4.0)), Ok(None))
        );
    }

    #[rstest]
    #[tokio::test(flavor = "multi_thread")]
    async fn a_database_that_stalls_fails_the_read_at_the_deadline_and_holds_no_later_one() {
        let (store, held) = stored().await;
        let stalled = held.locking("LiteLLM_EndUserTable").await;

        let while_stalled = spend(&store, "eu-blocked").await;
        stalled.rollback().await.unwrap();

        assert_eq!(
            (
                while_stalled,
                spend(&store, "eu-blocked").await,
                spend(&store, "eu-spent").await
            ),
            (Err(()), Ok(Some(0.0)), Ok(Some(3.0)))
        );
    }

    fn store(held: &Held) -> AuthStore {
        AuthStore::new(keys::memory_cache(), held.database())
    }

    #[rstest]
    #[tokio::test(flavor = "multi_thread")]
    async fn a_team_found_by_its_alias_is_cached_under_the_alias_and_its_id() {
        let held = Held::start().await;
        held.insert_named(
            "LiteLLM_TeamTable",
            &json!({"team_id": "team-1", "team_alias": "alpha"}),
        )
        .await;
        let store = store(&held);
        let id_of = |read: Result<AliasMatch<TeamEntry>, Error>| match read {
            Ok(AliasMatch::Found(team)) => Some(team.team_id),
            _ => None,
        };

        let from_database = id_of(store.team_by_alias("alpha").await);
        held.execute(r#"DELETE FROM "LiteLLM_TeamTable""#).await;
        let by_alias = id_of(store.team_by_alias("alpha").await);
        let by_id = store.team("team-1").await.unwrap();

        assert_eq!(
            (from_database, by_alias, by_id.map(|team| team.team_alias)),
            (
                Some("team-1".to_owned()),
                Some("team-1".to_owned()),
                Some(Some("alpha".to_owned()))
            )
        );
    }

    const JOINED: &str =
        r#"INSERT INTO "LiteLLM_TeamMembership" (user_id, team_id) VALUES ('user-1', 'team-1')"#;
    const LEFT: &str = r#"DELETE FROM "LiteLLM_TeamMembership""#;

    #[rstest]
    #[case::a_member(JOINED, LEFT, true)]
    #[case::no_member(LEFT, JOINED, false)]
    #[tokio::test(flavor = "multi_thread")]
    async fn a_membership_is_cached_whether_or_not_the_user_is_a_member(
        #[case] before: &'static str,
        #[case] after: &'static str,
        #[case] member: bool,
    ) {
        let held = Held::start().await;
        held.execute(before).await;
        let store = store(&held);
        let read = async || {
            let membership = store.team_membership("user-1", "team-1").await;
            membership.unwrap().is_some()
        };

        let from_database = read().await;
        held.execute(after).await;
        let cached = read().await;

        assert_eq!((from_database, cached), (member, member));
    }

    #[rstest]
    #[tokio::test(flavor = "multi_thread")]
    async fn a_user_read_from_the_database_is_cached() {
        let held = Held::start().await;
        held.insert_named("LiteLLM_UserTable", &json!({"user_id": "user-1"}))
            .await;
        let store = store(&held);
        let read = async || {
            let user = store.user("user-1").await.unwrap();
            user.map(|user| user.user_id)
        };

        let from_database = read().await;
        held.execute(r#"DELETE FROM "LiteLLM_UserTable""#).await;
        let cached = read().await;

        assert_eq!(
            (from_database, cached),
            (Some("user-1".to_owned()), Some("user-1".to_owned()))
        );
    }
}
