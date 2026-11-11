use std::collections::HashSet;

use litellm_secret_value::SecretValue;
use sha2::{Digest, Sha256};
use subtle::ConstantTimeEq;

use crate::{
    AuthHost, Error, KeyDenial, SessionDenial, key_budget,
    host::{EndUserRow, KeyRow, MembershipRow, Rows, TeamRow, UserRow},
    jwt::{IdentityRead, TeamRead, UserLookup, UserRead},
};

const UI_TEAM_ID: &str = "litellm-dashboard";
const USER_AUTH_ROUTE: &str = "/user/auth";
const VIRTUAL_KEY_PREFIX: &str = "sk-";

#[derive(Clone, Debug, Default)]
pub struct KeySettings {
    /// [`HostFacts::master_key`](crate::HostFacts::master_key)
    pub master_key: Option<SecretValue>,
    /// [`HostFacts::has_db`](crate::HostFacts::has_db)
    pub has_db: bool,
    /// [`general_settings.allow_user_auth`](litellm_config::GeneralSettings::allow_user_auth)
    pub allow_user_auth: bool,
    /// [`HostFacts::ui_login`](crate::HostFacts::ui_login)
    pub ui_login: bool,
    /// [`general_settings.enable_jwt_auth`](litellm_config::GeneralSettings::enable_jwt_auth), so a JWT
    /// sent while it is off gets a hint in the denial.
    pub jwt_auth: bool,
    /// [`HostFacts::proxy_budget`](crate::HostFacts::proxy_budget)
    pub proxy_budget: bool,
    /// [`MASTER_KEY_ONLY_ROUTES`](crate::routes::MASTER_KEY_ONLY_ROUTES)
    pub master_key_only_routes: HashSet<String>,
    /// `MINIMUM_CUSTOM_KEY_LENGTH`: shorter keys are never echoed back in an error, not even in part.
    pub min_custom_key_length: usize,
}

#[derive(Clone, Copy, Debug)]
pub struct KeyRequest<'a> {
    pub route: &'a str,
    pub token: Option<&'a str>,
    pub now: f64,
    /// `RouteChecks.is_llm_api_route(route)`, where the key's max budget applies.
    pub is_llm_api_route: bool,
}

/// The key fields the core decides on.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct KeyRecord {
    pub token: Option<String>,
    pub blocked: bool,
    pub expires: Option<f64>,
    pub is_proxy_admin: bool,
    pub is_session_token: bool,
    pub user_id: Option<String>,
    pub team_id: Option<String>,
    pub team_object_permission_id: Option<String>,
    pub project_id: Option<String>,
    pub last_refreshed_at: Option<f64>,
    /// The key owner's spend in the key's team.
    pub team_member_spend: Option<f64>,
    pub budget: crate::key_budget::KeyBudgetRecord,
}

/// A host's key row with the fields the core decides on.
#[derive(Clone, Debug, PartialEq)]
pub struct Keyed<K> {
    pub record: KeyRecord,
    pub row: K,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Owner<U> {
    /// Deactivated through SCIM.
    pub deactivated: bool,
    pub row: U,
}

#[derive(Clone, Debug, PartialEq)]
pub struct CachedTeam<T> {
    pub refreshed_at: Option<f64>,
    pub row: T,
}

/// The team values a virtual key carries into the request.
#[derive(Clone, Debug, PartialEq)]
pub enum TeamGrant<R: Rows> {
    NoTeam,
    Loaded(R::Team),
    /// The UI team, or a team `get_team_object` did not report: the key's own copy, with the
    /// object permission it names.
    KeyCopy(Option<R::ObjectPermission>),
}

#[derive(Clone, Debug, PartialEq)]
pub struct KeyContext<R: Rows> {
    pub key: R::Key,
    pub end_user: R::EndUser,
    pub owner: Option<R::User>,
    pub team: TeamGrant<R>,
    pub project: Option<R::Project>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum KeyIdentity<R: Rows> {
    /// No master key is configured, so any token, the one a pass-through endpoint read included, is let through.
    NoMasterKey {
        token: Option<String>,
    },
    UserAuthRoute,
    CachedAdmin {
        key: R::Key,
        end_user: R::EndUser,
    },
    /// `hash` is the master key's hash, which the host caches the admin identity under.
    MasterKey {
        hash: String,
        end_user: R::EndUser,
    },
    VirtualKey(Box<KeyContext<R>>),
}

/// An error, with the key the request had resolved before it, which the host labels its failure with.
#[derive(Clone, Debug, PartialEq)]
pub struct Refused<R: Rows> {
    pub error: Error,
    pub key: Option<R::Key>,
}

impl<R: Rows> From<Error> for Refused<R> {
    fn from(error: Error) -> Self {
        Self { error, key: None }
    }
}

pub type KeyOutcome<R> = Result<KeyIdentity<R>, Refused<R>>;

fn refused<R: Rows>(error: impl Into<Error>, key: Option<&R::Key>) -> Refused<R> {
    Refused {
        error: error.into(),
        key: key.cloned(),
    }
}

pub fn hash_token(token: &str) -> String {
    format!("{:x}", Sha256::digest(token.as_bytes()))
}

pub fn is_master_key(token: &str, master_key: &str) -> bool {
    token.is_ascii()
        && master_key.is_ascii()
        && bool::from(token.as_bytes().ct_eq(master_key.as_bytes()))
}

fn is_jwt(token: &str) -> bool {
    token.split('.').count() == 3
}

/// `abbreviate_api_key`
fn abbreviated(token: &str, settings: &KeySettings) -> String {
    let chars: Vec<char> = token.chars().collect();
    if chars.len() < settings.min_custom_key_length {
        return "sk-...".to_owned();
    }
    let last: String = chars[chars.len().saturating_sub(4)..].iter().collect();
    format!("sk-...{last}")
}

fn masked(token: &str) -> String {
    let chars: Vec<char> = token.chars().collect();
    if chars.len() <= 8 {
        return "****".to_owned();
    }
    let first: String = chars[..4].iter().collect();
    let last: String = chars[chars.len() - 4..].iter().collect();
    format!("{first}****{last}")
}

/// `shown` is the token the proxy names in the error: the presented one, or its hash once the key
/// was read from the database.
fn expired(key: &KeyRecord, now: f64, shown: &str, settings: &KeySettings) -> Option<KeyDenial> {
    key.expires
        .filter(|expires| *expires < now)
        .map(|expires| KeyDenial::Expired {
            expires,
            now,
            key: abbreviated(shown, settings),
        })
}

type Found<H> = Result<Option<Keyed<KeyRow<H>>>, Refused<<H as AuthHost>::Rows>>;

/// The current rows behind a session key's grants, or `Degraded` when one could not be read for a
/// reason unrelated to the caller, so the minted grants stand.
enum SessionRows<H: AuthHost> {
    Current {
        user: Option<UserRow<H>>,
        team: Option<TeamRow<H>>,
        membership: Option<MembershipRow<H>>,
    },
    Degraded,
}

/// `GrantResolver.resolve` for the session key's user and team. A user removed from its team,
/// or a user or team deleted outright, refuses the key.
async fn session_rows<H: AuthHost>(
    host: &H,
    record: &KeyRecord,
) -> Result<SessionRows<H>, SessionDenial> {
    let team_id = record.team_id.as_deref();
    let (user, membership) = match record.user_id.as_deref().filter(|id| !id.is_empty()) {
        None => (None, None),
        Some(user_id) => {
            let lookup = UserLookup {
                user_id,
                user_email: None,
                sso_user_id: None,
                upsert: false,
            };
            match host.resolve_user_identity(&lookup, team_id).await {
                Ok(IdentityRead {
                    user: UserRead::Found(user),
                    membership,
                }) => (Some(user), membership),
                Ok(IdentityRead {
                    user: UserRead::Absent,
                    ..
                }) => {
                    return Err(SessionDenial::UserGone {
                        user_id: user_id.to_owned(),
                    });
                }
                Err(_) => return Ok(SessionRows::Degraded),
            }
        }
    };
    let Some(team_id) = team_id else {
        return Ok(SessionRows::Current {
            user: user.map(|user| user.row),
            team: None,
            membership,
        });
    };
    if user
        .as_ref()
        .is_some_and(|user| !user.teams.iter().any(|team| team == team_id))
    {
        return Err(SessionDenial::NotAMember {
            team_id: team_id.to_owned(),
        });
    }
    let team = match host.team(team_id, false).await {
        Ok(TeamRead::Found(team)) => team.row,
        Ok(TeamRead::Absent) => {
            return Err(SessionDenial::TeamGone {
                team_id: team_id.to_owned(),
            });
        }
        Ok(TeamRead::Unreadable) | Err(_) => return Ok(SessionRows::Degraded),
    };
    Ok(SessionRows::Current {
        user: user.map(|user| user.row),
        team: Some(team),
        membership,
    })
}

async fn refreshed<H: AuthHost>(
    host: &H,
    key: Keyed<KeyRow<H>>,
    settings: &KeySettings,
) -> Found<H> {
    if !(key.record.is_session_token && settings.has_db) {
        return Ok(Some(key));
    }
    match session_rows(host, &key.record).await {
        Ok(SessionRows::Current {
            user: Some(user),
            team,
            membership,
        }) => host
            .with_session_grants(&key.row, &user, team.as_ref(), membership.as_ref())
            .await
            .map(Some)
            .map_err(|error| refused(error, Some(&key.row))),
        Ok(SessionRows::Current { user: None, .. } | SessionRows::Degraded) => Ok(Some(key)),
        Err(denial) => Err(refused(denial, Some(&key.row))),
    }
}

async fn found_key<H: AuthHost>(
    host: &H,
    token: &str,
    settings: &KeySettings,
    hash: &str,
) -> Found<H> {
    let cached = host.cached_key(hash).await.ok().flatten();
    let key = match cached {
        None if !token.starts_with(VIRTUAL_KEY_PREFIX) && settings.ui_login => {
            host.ui_key(token).await?
        }
        cached => cached,
    };
    match key {
        None => Ok(None),
        Some(key) => refreshed(host, key, settings).await,
    }
}

/// A key whose cached team was refreshed after it takes that team's values.
async fn with_cached_team<H: AuthHost>(
    host: &H,
    key: Keyed<KeyRow<H>>,
) -> Result<Keyed<KeyRow<H>>, Error> {
    let Some(team_id) = key
        .record
        .team_id
        .as_deref()
        .filter(|team_id| *team_id != UI_TEAM_ID)
    else {
        return Ok(key);
    };
    let Some(team) = host.cached_team(team_id).await.ok().flatten() else {
        return Ok(key);
    };
    match (team.refreshed_at, key.record.last_refreshed_at) {
        (Some(team_refreshed), Some(key_refreshed)) if team_refreshed > key_refreshed => {
            Ok(Keyed {
                row: host.with_team(&key.row, &team.row).await?,
                record: key.record,
            })
        }
        _ => Ok(key),
    }
}

pub async fn authenticate<H: AuthHost>(
    host: &H,
    request: &KeyRequest<'_>,
    settings: &KeySettings,
    resolved: Option<Keyed<KeyRow<H>>>,
) -> KeyOutcome<H::Rows> {
    let Some(master_key) = settings.master_key.as_ref() else {
        return Ok(KeyIdentity::NoMasterKey {
            token: request.token.map(str::to_owned),
        });
    };
    let Some(token) = request.token else {
        return Err(refused(KeyDenial::MissingKey, None));
    };
    if token.is_empty() {
        return Err(refused(KeyDenial::MalformedKey, None));
    }
    if request.route == USER_AUTH_ROUTE {
        return match settings.allow_user_auth {
            true => Ok(KeyIdentity::UserAuthRoute),
            false => Err(refused(KeyDenial::UserAuthDisabled, None)),
        };
    }
    let resolved_row = resolved.as_ref().map(|key| &key.row);
    let end_user = host
        .end_user(token, resolved_row)
        .await
        .map_err(|error| refused(error, resolved_row))?;
    let hash = hash_token(token);
    let key = match resolved {
        Some(key) => refreshed(host, key, settings).await?,
        None => found_key(host, token, settings, &hash).await?,
    };
    if let Some(key) = key.as_ref().filter(|key| key.record.is_proxy_admin) {
        if let Some(denial) = expired(&key.record, request.now, token, settings) {
            host.evict_key(&hash).await?;
            return Err(refused(denial, Some(&key.row)));
        }
        return Ok(KeyIdentity::CachedAdmin {
            key: key.row.clone(),
            end_user,
        });
    }
    let key = match key {
        Some(key) => Some(with_cached_team(host, key).await?),
        None => None,
    };
    if is_master_key(token, master_key.expose()) {
        return Ok(KeyIdentity::MasterKey { hash, end_user });
    }
    let row = key.as_ref().map(|key| &key.row);
    if settings.master_key_only_routes.contains(request.route) {
        let route = request.route.to_owned();
        return Err(refused(KeyDenial::MasterKeyOnlyRoute { route }, row));
    }
    if !settings.has_db {
        return Err(refused(KeyDenial::NoDb, row));
    }
    let (key, shown) = match key {
        Some(key) => (key, token),
        None if !token.starts_with(VIRTUAL_KEY_PREFIX) => {
            let denial = KeyDenial::NotVirtualKey {
                received: masked(token),
                jwt_hint: !settings.jwt_auth && is_jwt(token),
            };
            return Err(refused(denial, None));
        }
        None => match host.db_key(&hash, &end_user).await? {
            Some(key) => (key, hash.as_str()),
            None => return Err(refused(KeyDenial::NotFound, None)),
        },
    };
    validate(host, key, end_user, request, settings, shown).await
}

async fn key_team<H: AuthHost>(host: &H, record: &KeyRecord) -> Result<TeamGrant<H::Rows>, Error> {
    let loaded = match record.team_id.as_deref() {
        None => return Ok(TeamGrant::NoTeam),
        Some(UI_TEAM_ID) => None,
        Some(team_id) => host.key_team(team_id).await?,
    };
    if let Some(team) = loaded {
        return Ok(TeamGrant::Loaded(team));
    }
    let permission = match record.team_object_permission_id.as_deref() {
        None => None,
        Some(id) => host.object_permission(id).await?,
    };
    Ok(TeamGrant::KeyCopy(permission))
}

async fn validate<H: AuthHost>(
    host: &H,
    key: Keyed<KeyRow<H>>,
    end_user: EndUserRow<H>,
    request: &KeyRequest<'_>,
    settings: &KeySettings,
    shown: &str,
) -> KeyOutcome<H::Rows> {
    let record = key.record;
    if record.blocked {
        return Err(refused(KeyDenial::Blocked, Some(&key.row)));
    }
    let row = host.with_temp_budget(&key.row).await?;
    host.normalize_model(&row).await?;
    host.prefetch_auth_objects(&row, &end_user).await?;
    let owner = match record.user_id.as_deref() {
        None => None,
        Some(user_id) => match host.key_owner(user_id).await.ok().flatten() {
            Some(owner) if owner.deactivated => {
                let user_id = user_id.to_owned();
                return Err(refused(KeyDenial::UserDeactivated { user_id }, Some(&row)));
            }
            owner => owner.map(|owner| owner.row),
        },
    };
    key_budget::team_member_budget(host, &record, &row, owner.as_ref(), settings)
        .await
        .map_err(|error| refused(error, Some(&row)))?;
    if let Some(denial) = expired(&record, request.now, shown, settings) {
        return Err(refused(denial, Some(&row)));
    }
    key_budget::key_budgets(host, &record, &row, owner.as_ref(), request.is_llm_api_route)
        .await
        .map_err(|error| refused(error, Some(&row)))?;
    let team = key_team(host, &record).await?;
    let project = match record.project_id.as_deref() {
        None => None,
        Some(project_id) => host.project(project_id).await?,
    };
    key_budget::proxy_budget_alert(host, &row, settings).await?;
    if record.token.is_none() {
        return Err(refused(KeyDenial::NoToken, Some(&row)));
    }
    Ok(KeyIdentity::VirtualKey(Box::new(KeyContext {
        key: row,
        end_user,
        owner,
        team,
        project,
    })))
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use rstest::{fixture, rstest};

    use super::*;
    use crate::{
        Error,
        jwt::User,
        testing::{Labels, ScriptedHost},
    };

    const MASTER: &str = "sk-master";
    const NOW: f64 = 1_000.0;

    #[fixture]
    fn settings() -> KeySettings {
        KeySettings {
            master_key: Some(SecretValue::new(MASTER)),
            has_db: true,
            ui_login: true,
            master_key_only_routes: HashSet::from(["/global/spend/reset".to_owned()]),
            min_custom_key_length: 16,
            ..KeySettings::default()
        }
    }

    fn virtual_key() -> KeyRecord {
        KeyRecord {
            token: Some("hashed".to_owned()),
            ..KeyRecord::default()
        }
    }

    fn over_budget() -> crate::BudgetExceeded {
        crate::BudgetExceeded {
            current_cost: 2.0,
            max_budget: 1.0,
            message: "Budget has been exceeded!".to_owned(),
            entity_type: Some("key".to_owned()),
            entity_id: None,
        }
    }

    fn unavailable() -> Error {
        Error::Unavailable(crate::Unavailable {
            dependency: crate::Dependency::Db,
            transient: true,
            message: "db down".to_owned(),
        })
    }

    async fn run_with(
        host: &ScriptedHost,
        token: &str,
        route: &str,
        settings: &KeySettings,
    ) -> KeyOutcome<Labels> {
        let request = KeyRequest {
            route,
            token: Some(token),
            now: NOW,
            is_llm_api_route: route == "/chat/completions",
        };
        authenticate(host, &request, settings, None).await
    }

    /// The outcome with a denial's key dropped; `denials_carry_the_resolved_key` checks the key.
    async fn run(
        host: &ScriptedHost,
        token: &str,
        route: &str,
        settings: &KeySettings,
    ) -> Result<KeyIdentity<Labels>, Error> {
        run_with(host, token, route, settings)
            .await
            .map_err(|refused| refused.error)
    }

    fn context(outcome: Result<KeyIdentity<Labels>, Error>) -> KeyContext<Labels> {
        match outcome {
            Ok(KeyIdentity::VirtualKey(context)) => *context,
            other => panic!("expected a virtual key, got {other:?}"),
        }
    }

    #[test]
    fn hashes_with_sha256() {
        // FIPS 180-2 test vector for "abc"
        assert_eq!(
            hash_token("abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[rstest]
    #[case::equal(MASTER, MASTER, true)]
    #[case::different("sk-other", MASTER, false)]
    #[case::prefix("sk-mast", MASTER, false)]
    #[case::non_ascii_never_matches("sk-ü", "sk-ü", false)]
    fn master_key_comparison(#[case] token: &str, #[case] master: &str, #[case] expected: bool) {
        assert_eq!(is_master_key(token, master), expected);
    }

    #[rstest]
    #[tokio::test]
    async fn without_master_key_every_token_is_let_through(settings: KeySettings) {
        let settings = KeySettings {
            master_key: None,
            ..settings
        };
        let host = ScriptedHost::default();
        assert_eq!(
            run(&host, "", "/chat/completions", &settings).await,
            Ok(KeyIdentity::NoMasterKey {
                token: Some(String::new())
            })
        );
        assert!(host.calls().is_empty());
    }

    #[rstest]
    #[tokio::test]
    async fn missing_token_is_refused(settings: KeySettings) {
        let host = ScriptedHost::default();
        let request = KeyRequest {
            route: "/chat/completions",
            token: None,
            now: NOW,
            is_llm_api_route: true,
        };
        let outcome = authenticate(&host, &request, &settings, None).await;
        assert_eq!(
            outcome.map_err(|refused| refused.error),
            Err(KeyDenial::MissingKey.into())
        );
    }

    #[rstest]
    #[tokio::test]
    async fn empty_token_is_malformed(settings: KeySettings) {
        let host = ScriptedHost::default();
        assert_eq!(
            run(&host, "", "/chat/completions", &settings).await,
            Err(KeyDenial::MalformedKey.into())
        );
    }

    #[rstest]
    #[case::allowed(true, Ok(KeyIdentity::UserAuthRoute))]
    #[case::disabled(false, Err(KeyDenial::UserAuthDisabled.into()))]
    #[tokio::test]
    async fn user_auth_route(
        settings: KeySettings,
        #[case] allow_user_auth: bool,
        #[case] expected: Result<KeyIdentity<Labels>, Error>,
    ) {
        let settings = KeySettings {
            allow_user_auth,
            ..settings
        };
        let host = ScriptedHost::default();
        assert_eq!(run(&host, "sk-1", "/user/auth", &settings).await, expected);
        assert!(host.calls().is_empty());
    }

    #[rstest]
    #[tokio::test]
    async fn master_key_carries_the_end_user(settings: KeySettings) {
        let host = ScriptedHost::default();
        assert_eq!(
            run(&host, MASTER, "/chat/completions", &settings).await,
            Ok(KeyIdentity::MasterKey {
                hash: hash_token(MASTER),
                end_user: "end-user".to_owned()
            })
        );
        assert_eq!(host.calls(), ["end_user -", "cached_key"]);
    }

    #[rstest]
    #[tokio::test]
    async fn end_user_over_budget_is_refused_before_the_key_is_read(settings: KeySettings) {
        let host = ScriptedHost {
            end_user_over_budget: Some(over_budget()),
            ..ScriptedHost::default()
        };
        assert_eq!(
            run(&host, "sk-1", "/chat/completions", &settings).await,
            Err(Error::BudgetExceeded(over_budget()))
        );
        assert_eq!(host.calls(), ["end_user -"]);
    }

    #[rstest]
    #[case::fresh(None)]
    #[case::not_yet_expired(Some(NOW + 1.0))]
    #[tokio::test]
    async fn cached_admin_key_carries_its_row(settings: KeySettings, #[case] expires: Option<f64>) {
        let host = ScriptedHost {
            cached: Some(KeyRecord {
                is_proxy_admin: true,
                expires,
                ..virtual_key()
            }),
            ..ScriptedHost::default()
        };
        assert_eq!(
            run(&host, "sk-admin", "/chat/completions", &settings).await,
            Ok(KeyIdentity::CachedAdmin {
                key: "cached".to_owned(),
                end_user: "end-user".to_owned()
            })
        );
    }

    #[rstest]
    #[tokio::test]
    async fn expired_cached_admin_key_is_evicted_then_refused(settings: KeySettings) {
        let host = ScriptedHost {
            cached: Some(KeyRecord {
                is_proxy_admin: true,
                expires: Some(NOW - 1.0),
                ..virtual_key()
            }),
            ..ScriptedHost::default()
        };
        assert_eq!(
            run(&host, "abc", "/chat/completions", &settings).await,
            Err(KeyDenial::Expired {
                expires: NOW - 1.0,
                now: NOW,
                key: "sk-...".to_owned(),
            }
            .into())
        );
        assert_eq!(
            host.calls().last().unwrap(),
            "evict ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[rstest]
    #[tokio::test]
    async fn master_key_only_route_refuses_virtual_keys(settings: KeySettings) {
        let host = ScriptedHost::default();
        assert_eq!(
            run(&host, "sk-1", "/global/spend/reset", &settings).await,
            Err(KeyDenial::MasterKeyOnlyRoute {
                route: "/global/spend/reset".to_owned()
            }
            .into())
        );
        assert!(matches!(
            run(&host, MASTER, "/global/spend/reset", &settings).await,
            Ok(KeyIdentity::MasterKey { .. })
        ));
    }

    #[rstest]
    #[tokio::test]
    async fn without_db_only_the_master_key_works(settings: KeySettings) {
        let settings = KeySettings {
            has_db: false,
            ..settings
        };
        let host = ScriptedHost::default();
        assert_eq!(
            run(&host, "sk-1", "/chat/completions", &settings).await,
            Err(KeyDenial::NoDb.into())
        );
    }

    #[rstest]
    #[case::random("not-a-key", false, Err(KeyDenial::NotVirtualKey { received: "not-****-key".to_owned(), jwt_hint: false }.into()))]
    #[case::jwt_with_jwt_auth_off("a.b.c", false, Err(KeyDenial::NotVirtualKey { received: "****".to_owned(), jwt_hint: true }.into()))]
    #[case::jwt_with_jwt_auth_on("a.b.c", true, Err(KeyDenial::NotVirtualKey { received: "****".to_owned(), jwt_hint: false }.into()))]
    #[tokio::test]
    async fn non_virtual_key_tokens(
        settings: KeySettings,
        #[case] token: &str,
        #[case] jwt_auth: bool,
        #[case] expected: Result<KeyIdentity<Labels>, Error>,
    ) {
        let settings = KeySettings {
            jwt_auth,
            ui_login: false,
            ..settings
        };
        let host = ScriptedHost::default();
        assert_eq!(
            run(&host, token, "/chat/completions", &settings).await,
            expected
        );
        assert_eq!(host.calls(), ["end_user -", "cached_key"]);
    }

    #[rstest]
    #[tokio::test]
    async fn virtual_key_missing_from_the_database_is_not_found(settings: KeySettings) {
        let host = ScriptedHost::default();
        assert_eq!(
            run(&host, "sk-1", "/chat/completions", &settings).await,
            Err(KeyDenial::NotFound.into())
        );
        assert_eq!(
            host.calls(),
            ["end_user -", "cached_key", "db_key end-user"]
        );
    }

    #[rstest]
    #[case::non_virtual_key_tries_ui_login("ui-session-blob", true, &["end_user -", "cached_key", "ui_key"])]
    #[case::ui_login_off("ui-session-blob", false, &["end_user -", "cached_key"])]
    #[case::virtual_key_skips_ui_login("sk-1", true, &["end_user -", "cached_key", "db_key end-user"])]
    #[tokio::test]
    async fn ui_login_key_is_only_tried_for_non_virtual_keys(
        settings: KeySettings,
        #[case] token: &str,
        #[case] ui_login: bool,
        #[case] expected_prefix: &[&str],
    ) {
        let settings = KeySettings {
            ui_login,
            ..settings
        };
        let host = ScriptedHost::default();
        let _ = run(&host, token, "/chat/completions", &settings).await;
        assert_eq!(&host.calls()[..expected_prefix.len()], expected_prefix);
    }

    #[rstest]
    #[tokio::test]
    async fn virtual_key_runs_every_check_in_order_and_keeps_every_row(settings: KeySettings) {
        let settings = KeySettings {
            proxy_budget: true,
            ..settings
        };
        let host = ScriptedHost {
            db: Some(KeyRecord {
                user_id: Some("u".to_owned()),
                team_id: Some("t".to_owned()),
                project_id: Some("p".to_owned()),
                ..virtual_key()
            }),
            key_owner: Some(false),
            key_team_found: true,
            ..ScriptedHost::default()
        };
        assert_eq!(
            context(run(&host, "sk-1", "/chat/completions", &settings).await),
            KeyContext {
                key: "db+prepared".to_owned(),
                end_user: "end-user".to_owned(),
                owner: Some("owner u".to_owned()),
                team: TeamGrant::Loaded("team t".to_owned()),
                project: Some("project p".to_owned()),
            }
        );
        assert_eq!(
            host.calls(),
            [
                "end_user -",
                "cached_key",
                "db_key end-user",
                "temp_budget db",
                "normalize db+prepared",
                "prefetch db+prepared end-user",
                "key_owner u",
                "key_skips_budget_checks db+prepared",
                "key_model_budgets db+prepared owner u",
                "key_team t",
                "project p",
                "global_proxy_spend",
            ]
        );
    }

    #[rstest]
    #[tokio::test]
    async fn blocked_key_is_refused_before_any_request_work(settings: KeySettings) {
        let host = ScriptedHost {
            cached: Some(KeyRecord {
                blocked: true,
                ..virtual_key()
            }),
            ..ScriptedHost::default()
        };
        assert_eq!(
            run(&host, "sk-1", "/chat/completions", &settings).await,
            Err(KeyDenial::Blocked.into())
        );
        assert_eq!(host.calls(), ["end_user -", "cached_key"]);
    }

    #[rstest]
    #[tokio::test]
    async fn deactivated_owner_is_refused(settings: KeySettings) {
        let host = ScriptedHost {
            cached: Some(KeyRecord {
                user_id: Some("u".to_owned()),
                ..virtual_key()
            }),
            key_owner: Some(true),
            ..ScriptedHost::default()
        };
        assert_eq!(
            run(&host, "sk-1", "/chat/completions", &settings).await,
            Err(KeyDenial::UserDeactivated {
                user_id: "u".to_owned()
            }
            .into())
        );
        assert_eq!(host.calls().last().unwrap(), "key_owner u");
    }

    #[rstest]
    #[tokio::test]
    async fn the_key_budget_step_denies_with_what_the_host_reported(settings: KeySettings) {
        let host = ScriptedHost {
            cached: Some(virtual_key()),
            key_over_budget: Some(over_budget()),
            ..ScriptedHost::default()
        };
        assert_eq!(
            run(&host, "sk-1", "/chat/completions", &settings).await,
            Err(Error::BudgetExceeded(over_budget()))
        );
        assert_eq!(host.calls().last().unwrap(), "key_model_budgets cached+prepared -");
    }

    #[rstest]
    #[tokio::test]
    async fn expiry_is_checked_after_the_team_member_budget(settings: KeySettings) {
        let host = ScriptedHost {
            cached: Some(KeyRecord {
                expires: Some(NOW - 1.0),
                user_id: Some("u".to_owned()),
                team_id: Some("t".to_owned()),
                team_member_spend: Some(0.0),
                ..virtual_key()
            }),
            ..ScriptedHost::default()
        };
        assert_eq!(
            run(&host, "sk-1", "/chat/completions", &settings).await,
            Err(KeyDenial::Expired {
                expires: NOW - 1.0,
                now: NOW,
                key: "sk-...".to_owned(),
            }
            .into())
        );
        assert!(
            host.calls()
                .last()
                .unwrap()
                .starts_with("team_member_max_budget")
        );
    }

    #[rstest]
    #[case::no_team(None, None, true, TeamGrant::NoTeam)]
    #[case::ui_team(Some("litellm-dashboard"), None, true, TeamGrant::KeyCopy(None))]
    #[case::unresolvable_team(Some("t"), None, false, TeamGrant::KeyCopy(None))]
    #[case::unresolvable_team_with_permission(Some("t"), Some("op"), false, TeamGrant::KeyCopy(Some("permission op".to_owned())))]
    #[case::resolved_team(Some("t"), Some("op"), true, TeamGrant::Loaded("team t".to_owned()))]
    #[tokio::test]
    async fn team_falls_back_to_the_key_copy(
        settings: KeySettings,
        #[case] team_id: Option<&str>,
        #[case] team_object_permission_id: Option<&str>,
        #[case] key_team_found: bool,
        #[case] expected: TeamGrant<Labels>,
    ) {
        let host = ScriptedHost {
            cached: Some(KeyRecord {
                team_id: team_id.map(str::to_owned),
                team_object_permission_id: team_object_permission_id.map(str::to_owned),
                ..virtual_key()
            }),
            key_team_found,
            ..ScriptedHost::default()
        };
        assert_eq!(
            context(run(&host, "sk-1", "/chat/completions", &settings).await).team,
            expected
        );
    }

    #[rstest]
    #[tokio::test]
    async fn key_without_token_is_refused_last(settings: KeySettings) {
        let host = ScriptedHost {
            cached: Some(KeyRecord::default()),
            ..ScriptedHost::default()
        };
        assert_eq!(
            run(&host, "sk-1", "/chat/completions", &settings).await,
            Err(KeyDenial::NoToken.into())
        );
    }

    fn member_of(teams: &[&str]) -> User<String> {
        User {
            user_id: "u".to_owned(),
            teams: teams.iter().map(|team| (*team).to_owned()).collect(),
            row: "user u".to_owned(),
            ..User::default()
        }
    }

    fn session_key(team_id: Option<&str>) -> KeyRecord {
        KeyRecord {
            is_session_token: true,
            user_id: Some("u".to_owned()),
            team_id: team_id.map(str::to_owned),
            ..virtual_key()
        }
    }

    fn found_team(team_id: &str) -> Result<TeamRead<String>, Error> {
        Ok(TeamRead::Found(crate::jwt::Team {
            team_id: team_id.to_owned(),
            row: format!("team {team_id}"),
            ..crate::jwt::Team::default()
        }))
    }

    #[rstest]
    #[case::session_key_without_team(
        true,
        None,
        "refreshed+prepared",
        &["resolve_user u team=-", "with_session_grants cached user u - -"]
    )]
    #[case::session_key_in_its_team(
        true,
        Some("t"),
        "refreshed+prepared",
        &["resolve_user u team=t", "team t", "with_session_grants cached user u team t membership u t"]
    )]
    #[case::plain_key(false, Some("t"), "cached+prepared", &[])]
    #[tokio::test]
    async fn session_tokens_take_the_grants_of_their_current_rows(
        settings: KeySettings,
        #[case] is_session_token: bool,
        #[case] team_id: Option<&str>,
        #[case] key: &str,
        #[case] refresh_calls: &[&str],
    ) {
        let host = ScriptedHost {
            cached: Some(KeyRecord {
                is_session_token,
                ..session_key(team_id)
            }),
            refreshed: virtual_key(),
            users: HashMap::from([("u", member_of(&["t"]))]),
            teams: HashMap::from([("t", found_team("t"))]),
            ..ScriptedHost::default()
        };
        assert_eq!(
            context(run(&host, "sk-1", "/chat/completions", &settings).await).key,
            key
        );
        let calls = host.calls();
        let refreshes: Vec<&str> = calls
            .iter()
            .map(String::as_str)
            .filter(|call| {
                call.starts_with("resolve_user")
                    || call.starts_with("team ")
                    || call.starts_with("with_session_grants")
            })
            .collect();
        assert_eq!(refreshes, refresh_calls);
    }

    #[rstest]
    #[case::user_gone(&[], &[], crate::SessionDenial::UserGone { user_id: "u".to_owned() })]
    #[case::not_a_member(&["u"], &["other"], crate::SessionDenial::NotAMember { team_id: "t".to_owned() })]
    #[case::team_gone(&["u"], &["t"], crate::SessionDenial::TeamGone { team_id: "t".to_owned() })]
    #[tokio::test]
    async fn session_refresh_refuses_a_key_whose_user_or_team_is_gone(
        settings: KeySettings,
        #[case] users: &[&'static str],
        #[case] user_teams: &[&str],
        #[case] denial: crate::SessionDenial,
    ) {
        let host = ScriptedHost {
            cached: Some(session_key(Some("t"))),
            users: users.iter().map(|id| (*id, member_of(user_teams))).collect(),
            ..ScriptedHost::default()
        };
        assert_eq!(
            run(&host, "sk-1", "/chat/completions", &settings).await,
            Err(KeyDenial::Session(denial).into())
        );
    }

    #[rstest]
    #[case::team_unreadable(Ok(TeamRead::Unreadable))]
    #[case::team_read_fails(Err(Error::HostFailed))]
    #[tokio::test]
    async fn an_unreadable_team_keeps_the_minted_grants(
        settings: KeySettings,
        #[case] team: Result<TeamRead<String>, Error>,
    ) {
        let host = ScriptedHost {
            cached: Some(session_key(Some("t"))),
            users: HashMap::from([("u", member_of(&["t"]))]),
            teams: HashMap::from([("t", team)]),
            ..ScriptedHost::default()
        };
        assert_eq!(
            context(run(&host, "sk-1", "/chat/completions", &settings).await).key,
            "cached+prepared"
        );
    }

    #[rstest]
    #[case::team_newer(Some(2.0), Some(1.0), "cached+team+prepared")]
    #[case::team_older(Some(1.0), Some(2.0), "cached+prepared")]
    #[case::key_never_refreshed(Some(2.0), None, "cached+prepared")]
    #[case::team_never_refreshed(None, Some(1.0), "cached+prepared")]
    #[tokio::test]
    async fn cached_team_values_reach_a_stale_key(
        settings: KeySettings,
        #[case] team_refreshed_at: Option<f64>,
        #[case] key_refreshed_at: Option<f64>,
        #[case] key: &str,
    ) {
        let host = ScriptedHost {
            cached: Some(KeyRecord {
                team_id: Some("t".to_owned()),
                last_refreshed_at: key_refreshed_at,
                ..virtual_key()
            }),
            team_refreshed_at,
            key_team_found: true,
            ..ScriptedHost::default()
        };
        assert_eq!(
            context(run(&host, "sk-1", "/chat/completions", &settings).await).key,
            key
        );
    }

    #[rstest]
    #[tokio::test]
    async fn failed_cache_read_falls_through_to_the_database(settings: KeySettings) {
        let host = ScriptedHost {
            cached: Some(virtual_key()),
            failing: HashMap::from([("cached_key", Error::HostFailed)]),
            ..ScriptedHost::default()
        };
        assert_eq!(
            run(&host, "sk-1", "/chat/completions", &settings).await,
            Err(KeyDenial::NotFound.into())
        );
    }

    #[rstest]
    #[case::cached_team("cached_team", None, "cached+prepared")]
    #[case::deactivated_owner("key_owner", Some(true), "cached+team+prepared")]
    #[tokio::test]
    async fn failed_lookups_are_treated_as_absent(
        settings: KeySettings,
        #[case] failing: &'static str,
        #[case] key_owner: Option<bool>,
        #[case] key: &str,
    ) {
        let host = ScriptedHost {
            cached: Some(KeyRecord {
                user_id: Some("u".to_owned()),
                team_id: Some("t".to_owned()),
                last_refreshed_at: Some(1.0),
                ..virtual_key()
            }),
            team_refreshed_at: Some(2.0),
            key_owner,
            failing: HashMap::from([(failing, Error::HostFailed)]),
            ..ScriptedHost::default()
        };
        let context = context(run(&host, "sk-1", "/chat/completions", &settings).await);
        assert_eq!((context.key.as_str(), context.owner), (key, None));
    }

    #[rstest]
    #[case::end_user("end_user")]
    #[case::db_key("db_key")]
    #[case::key_team("key_team")]
    #[case::project("project")]
    #[tokio::test]
    async fn other_host_failures_end_the_call(
        settings: KeySettings,
        #[case] failing: &'static str,
    ) {
        let host = ScriptedHost {
            db: Some(KeyRecord {
                team_id: Some("t".to_owned()),
                project_id: Some("p".to_owned()),
                ..virtual_key()
            }),
            failing: HashMap::from([(failing, unavailable())]),
            ..ScriptedHost::default()
        };
        assert_eq!(
            run(&host, "sk-1", "/chat/completions", &settings).await,
            Err(unavailable())
        );
    }

    #[rstest]
    #[tokio::test]
    async fn resolved_key_skips_the_lookups(settings: KeySettings) {
        let host = ScriptedHost::default();
        let request = KeyRequest {
            route: "/chat/completions",
            token: Some("hashed"),
            now: NOW,
            is_llm_api_route: true,
        };
        let resolved = Keyed {
            record: virtual_key(),
            row: "mapped".to_owned(),
        };
        let outcome = authenticate(&host, &request, &settings, Some(resolved)).await;
        assert_eq!(
            context(outcome.map_err(|refused| refused.error)).key,
            "mapped+prepared"
        );
        assert_eq!(
            &host.calls()[..4],
            [
                "end_user mapped",
                "temp_budget mapped",
                "normalize mapped+prepared",
                "prefetch mapped+prepared end-user"
            ]
        );
    }

    #[rstest]
    #[case::not_found_has_no_key("sk-1", None, None)]
    #[case::blocked_cached_key("sk-1", Some(KeyRecord { blocked: true, ..virtual_key() }), Some("cached"))]
    #[case::expired_key_after_prepare("sk-1", Some(KeyRecord { expires: Some(NOW - 1.0), ..virtual_key() }), Some("cached+prepared"))]
    #[tokio::test]
    async fn denials_carry_the_resolved_key(
        settings: KeySettings,
        #[case] token: &str,
        #[case] cached: Option<KeyRecord>,
        #[case] key: Option<&str>,
    ) {
        let host = ScriptedHost {
            cached,
            ..ScriptedHost::default()
        };
        let refused = run_with(&host, token, "/chat/completions", &settings)
            .await
            .unwrap_err();
        assert_eq!(refused.key.as_deref(), key);
    }

    #[rstest]
    #[case::short_key_is_hidden("sk-1", 16, "sk-...")]
    #[case::long_key_shows_its_last_four("sk-1234567890abcdef", 16, "sk-...cdef")]
    fn expired_key_names_only_the_tail(
        #[case] token: &str,
        #[case] min_custom_key_length: usize,
        #[case] expected: &str,
    ) {
        let settings = KeySettings {
            min_custom_key_length,
            ..KeySettings::default()
        };
        assert_eq!(abbreviated(token, &settings), expected);
    }
}
