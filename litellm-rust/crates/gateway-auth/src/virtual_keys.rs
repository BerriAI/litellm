use std::{
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
    time::{Duration, Instant},
};

use axum::http::request::Parts;
use moka::Expiry;

use crate::{
    AuthFuture, Authentication, AuthenticationMethod, Authenticator, Clock, Credential, Error,
    KeyError, KeyGrant, Permissions, Principal, PrincipalKind, VerifiedIdentity,
    keys::{KeyEntry, KeyHash, KeyLookup, KeyStatus, LookupFuture, verify_key},
};

pub(crate) const AUTHORITY: &str = "litellm:virtual-key";

#[derive(Clone, Debug)]
pub struct KeyCacheSettings {
    /// How long a found key (active, blocked or revoked) is trusted before it is read again.
    pub positive_ttl: Duration,
    /// How long an unknown key is remembered, and how long a failed lookup pauses reads.
    pub negative_ttl: Duration,
    /// How long the last answer for a key is kept to serve while the store is unreachable.
    pub max_stale: Duration,
    /// Upper bound on a single lookup before it counts as a store failure.
    pub lookup_timeout: Duration,
    pub capacity: u64,
}

impl Default for KeyCacheSettings {
    fn default() -> Self {
        Self {
            positive_ttl: Duration::from_secs(60),
            negative_ttl: Duration::from_secs(5),
            max_stale: Duration::from_secs(60 * 60),
            lookup_timeout: Duration::from_secs(5),
            capacity: 1_000_000,
        }
    }
}

#[derive(Clone)]
struct Cached {
    entry: KeyEntry,
    ttl: Duration,
}

struct PerEntryTtl;

impl Expiry<KeyHash, Cached> for PerEntryTtl {
    fn expire_after_create(&self, _: &KeyHash, value: &Cached, _: Instant) -> Option<Duration> {
        Some(value.ttl)
    }
}

/// A [`KeyLookup`] cache in front of the key store.
///
/// Fresh answers live for `positive_ttl` (found) or `negative_ttl` (unknown), and concurrent misses
/// for one key share a single store read. The last answer per key is also kept for `max_stale`:
/// when the store fails, that answer is served and further reads pause for `negative_ttl`, so keys
/// seen recently keep working through an outage and requests do not queue on a dead store.
///
/// [`CachedKeyLookup::invalidate`] drops answers the proxy reports changed, so a key revoked or
/// edited there stops being served from here before its TTL runs out.
pub struct CachedKeyLookup {
    store: Arc<dyn KeyLookup>,
    fresh: moka::future::Cache<KeyHash, Cached>,
    stale: moka::sync::Cache<KeyHash, KeyEntry>,
    settings: KeyCacheSettings,
    started: Instant,
    /// Milliseconds after `started` until which the store is not read; 0 when healthy.
    paused_until_ms: AtomicU64,
    /// Bumped by every invalidation, so a store read that overlapped one is not cached: it may
    /// have read the row before the change it reports.
    generation: AtomicU64,
}

impl CachedKeyLookup {
    pub fn new(store: Arc<dyn KeyLookup>, settings: KeyCacheSettings) -> Self {
        Self {
            store,
            fresh: moka::future::Cache::builder()
                .max_capacity(settings.capacity)
                .expire_after(PerEntryTtl)
                .support_invalidation_closures()
                .build(),
            stale: moka::sync::Cache::builder()
                .max_capacity(settings.capacity)
                .time_to_live(settings.max_stale)
                .support_invalidation_closures()
                .build(),
            settings,
            started: Instant::now(),
            paused_until_ms: AtomicU64::new(0),
            generation: AtomicU64::new(0),
        }
    }

    /// Drops the cached answers that a proxy auth cache invalidation for `cache_key` makes
    /// stale: a key's hash drops that key, `team_id:<id>` every key of the team, and
    /// `team_membership:<user>:<team>` or a bare user id every key of the user. The last known
    /// answer is dropped too, so an outage cannot bring a revoked key back.
    pub async fn invalidate(&self, cache_key: &str) {
        self.generation.fetch_add(1, Ordering::Relaxed);
        if let Some(hash) = KeyHash::from_hashed(cache_key) {
            self.fresh.invalidate(&hash).await;
            self.stale.invalidate(&hash);
            return;
        }
        let owner = match cache_key.strip_prefix("team_id:") {
            Some(team_id) => Owner::Team(team_id.to_owned()),
            None => Owner::User(match cache_key.strip_prefix("team_membership:") {
                Some(membership) => membership
                    .rsplit_once(':')
                    .map_or(membership, |(user_id, _)| user_id)
                    .to_owned(),
                None => cache_key.to_owned(),
            }),
        };
        let stale_owner = owner.clone();
        // Fails only without invalidation closures, which `new` enables.
        let _ = self
            .fresh
            .invalidate_entries_if(move |_, cached| owner.owns(&cached.entry));
        let _ = self
            .stale
            .invalidate_entries_if(move |_, entry| stale_owner.owns(entry));
    }

    /// Drops every fresh answer, for when invalidations may have been missed. The last known
    /// answers stay to serve through a store outage.
    pub fn invalidate_all(&self) {
        self.generation.fetch_add(1, Ordering::Relaxed);
        self.fresh.invalidate_all();
    }

    fn elapsed_ms(&self) -> u64 {
        u64::try_from(self.started.elapsed().as_millis()).unwrap_or(u64::MAX)
    }

    fn paused(&self) -> bool {
        self.elapsed_ms() < self.paused_until_ms.load(Ordering::Relaxed)
    }

    fn pause(&self) {
        let until = self.elapsed_ms().saturating_add(
            u64::try_from(self.settings.negative_ttl.as_millis()).unwrap_or(u64::MAX),
        );
        self.paused_until_ms.store(until, Ordering::Relaxed);
    }

    fn ttl(&self, status: &Option<KeyStatus>) -> Duration {
        match status {
            Some(_) => self.settings.positive_ttl,
            None => self.settings.negative_ttl,
        }
    }

    async fn read_store(&self, hash: &KeyHash) -> Result<Cached, KeyError> {
        let generation = self.generation.load(Ordering::Relaxed);
        let result =
            tokio::time::timeout(self.settings.lookup_timeout, self.store.lookup_entry(hash))
                .await
                .unwrap_or_else(|elapsed| Err(KeyError::Lookup(Box::new(elapsed))));
        match result {
            Ok(entry) => {
                self.paused_until_ms.store(0, Ordering::Relaxed);
                let ttl = match self.generation.load(Ordering::Relaxed) == generation {
                    true => self.ttl(&entry.status),
                    false => Duration::ZERO,
                };
                if !ttl.is_zero() {
                    self.stale.insert(hash.clone(), entry.clone());
                }
                Ok(Cached { entry, ttl })
            }
            Err(error) => {
                self.pause();
                match self.stale.get(hash) {
                    Some(entry) => {
                        tracing::warn!(
                            error = %error,
                            "virtual key store unavailable, serving the last known answer"
                        );
                        // Cache it briefly so the next read retries the store soon after it returns.
                        Ok(Cached {
                            entry,
                            ttl: self.settings.negative_ttl,
                        })
                    }
                    None => Err(error),
                }
            }
        }
    }
}

/// Whose change an invalidation reports.
#[derive(Clone)]
enum Owner {
    Team(String),
    User(String),
}

impl Owner {
    fn owns(&self, entry: &KeyEntry) -> bool {
        match self {
            Self::Team(team_id) => entry.team_id.as_ref() == Some(team_id),
            Self::User(user_id) => entry.user_id.as_ref() == Some(user_id),
        }
    }
}

impl KeyLookup for CachedKeyLookup {
    fn lookup<'a>(&'a self, hash: &'a KeyHash) -> LookupFuture<'a> {
        Box::pin(async move {
            if let Some(cached) = self.fresh.get(hash).await {
                return Ok(cached.entry.status);
            }
            if self.paused() {
                return match self.stale.get(hash) {
                    Some(entry) => Ok(entry.status),
                    None => Err(KeyError::Lookup("virtual key store unavailable".into())),
                };
            }
            self.fresh
                .try_get_with_by_ref(hash, self.read_store(hash))
                .await
                .map(|cached| cached.entry.status)
                .map_err(|error| KeyError::Lookup(Box::new(error)))
        })
    }
}

/// Accepts the configured credential (normally the master key) and, failing that, virtual keys.
pub struct VirtualKeyAuthenticator {
    primary: Arc<dyn Authenticator>,
    keys: Arc<CachedKeyLookup>,
    clock: Arc<dyn Clock>,
}

impl VirtualKeyAuthenticator {
    pub fn new(
        primary: Arc<dyn Authenticator>,
        keys: Arc<CachedKeyLookup>,
        clock: Arc<dyn Clock>,
    ) -> Self {
        Self {
            primary,
            keys,
            clock,
        }
    }
}

impl Authenticator for VirtualKeyAuthenticator {
    fn validate(&self) -> AuthFuture<'_, ()> {
        self.primary.validate()
    }

    fn verify<'a>(
        &'a self,
        credential: &'a Credential,
        request: &'a Parts,
    ) -> AuthFuture<'a, VerifiedIdentity> {
        Box::pin(async move {
            match self.primary.verify(credential, request).await {
                Err(Error::InvalidToken | Error::Unconfigured) => (),
                verified => return verified,
            }
            let Credential::Token(token) = credential else {
                return Err(Error::InvalidToken);
            };
            let key = verify_key(self.keys.as_ref(), token, || self.clock.now()).await?;
            let id = key.hash.as_str().to_owned();
            Ok(VerifiedIdentity {
                principal: Principal::new(AUTHORITY.into(), id.clone(), PrincipalKind::Service),
                authentication: Authentication {
                    method: AuthenticationMethod::VirtualKey,
                    verifier: AUTHORITY.into(),
                    credential_id: id,
                    expires_at: key.expires_at,
                },
                restrictions: Permissions::VirtualKey(KeyGrant {
                    models: key.models,
                    routes: key.allowed_routes,
                    team_id: key.team_id,
                }),
            })
        })
    }
}

#[cfg(test)]
mod tests {
    use std::{
        collections::HashMap,
        sync::{
            Mutex,
            atomic::{AtomicBool, AtomicUsize},
        },
        time::SystemTime,
    };

    use axum::http::Request;
    use litellm_auth_types::SecretValue;
    use rstest::{fixture, rstest};

    use super::*;
    use crate::{
        AccessRequest, Auth, LocalAdministrator, NoAdditionalPolicy, SystemClock,
        keys::{AllowedRoutes, KeyModels, ModelAccess, ModelDenial, Owner, Refusal, Restriction},
    };

    #[derive(Default)]
    struct Store {
        keys: Mutex<HashMap<KeyHash, KeyStatus>>,
        reads: AtomicUsize,
        down: AtomicBool,
    }

    impl Store {
        fn put(&self, token: &str, status: KeyStatus) {
            self.keys
                .lock()
                .unwrap()
                .insert(KeyHash::from_token(&SecretValue::new(token)), status);
        }
    }

    impl KeyLookup for Store {
        fn lookup<'a>(&'a self, hash: &'a KeyHash) -> LookupFuture<'a> {
            Box::pin(async move {
                self.reads.fetch_add(1, Ordering::SeqCst);
                if self.down.load(Ordering::SeqCst) {
                    return Err(KeyError::Lookup("connection refused".into()));
                }
                Ok(self.keys.lock().unwrap().get(hash).cloned())
            })
        }
    }

    struct Master;

    impl Authenticator for Master {
        fn verify<'a>(
            &'a self,
            credential: &'a Credential,
            _: &'a Parts,
        ) -> AuthFuture<'a, VerifiedIdentity> {
            Box::pin(async move {
                match credential {
                    Credential::Token(token) if token.expose() == "sk-master" => {
                        Ok(VerifiedIdentity {
                            principal: Principal::new(
                                "litellm:master-key".into(),
                                "master".into(),
                                PrincipalKind::System,
                            ),
                            authentication: Authentication {
                                method: AuthenticationMethod::MasterKey,
                                verifier: "litellm:master-key".into(),
                                credential_id: "master".into(),
                                expires_at: None,
                            },
                            restrictions: Permissions::All,
                        })
                    }
                    _ => Err(Error::InvalidToken),
                }
            })
        }
    }

    fn settings() -> KeyCacheSettings {
        KeyCacheSettings {
            positive_ttl: Duration::from_millis(200),
            negative_ttl: Duration::from_millis(100),
            ..KeyCacheSettings::default()
        }
    }

    fn strings(values: &[&str]) -> Vec<String> {
        values.iter().map(|value| (*value).to_owned()).collect()
    }

    fn active(models: &[&str], expires_at: Option<SystemTime>) -> KeyStatus {
        KeyStatus::active(expires_at, ModelAccess::new(strings(models)))
    }

    #[fixture]
    fn store() -> Arc<Store> {
        let store = Store::default();
        store.put("sk-all", active(&[], None));
        store.put("sk-scoped", active(&["gpt-a", "claude-*"], None));
        store.put(
            "sk-expired",
            active(&[], Some(SystemTime::now() - Duration::from_secs(1))),
        );
        store.put(
            "sk-later",
            active(&[], Some(SystemTime::now() + Duration::from_secs(3600))),
        );
        store.put("sk-blocked", KeyStatus::Blocked);
        store.put(
            "sk-team",
            KeyStatus::Active {
                expires_at: None,
                models: KeyModels {
                    key: ModelAccess::new(strings(&["gpt-a", "gpt-b"])),
                    team: Some(ModelAccess::new(strings(&["gpt-a"]))),
                },
                allowed_routes: AllowedRoutes::default(),
                team_id: Some("team-1".into()),
            },
        );
        store.put(
            "sk-anthropic-only",
            KeyStatus::Active {
                expires_at: None,
                models: KeyModels::default(),
                allowed_routes: AllowedRoutes::new(strings(&["anthropic_routes"])),
                team_id: None,
            },
        );
        store.put(
            "sk-over-budget",
            KeyStatus::Refused {
                expires_at: None,
                refusal: Refusal::BudgetExceeded {
                    owner: "Key=key".into(),
                    spend: "2.0".into(),
                    max_budget: "1.0".into(),
                },
            },
        );
        store.put(
            "sk-rate-limited",
            KeyStatus::Refused {
                expires_at: None,
                refusal: Refusal::Unsupported {
                    restriction: Restriction::RateLimits,
                    owner: Owner::Key,
                },
            },
        );
        store.put(
            "sk-team-blocked",
            KeyStatus::Refused {
                expires_at: None,
                refusal: Refusal::TeamBlocked {
                    team_id: "team-1".into(),
                },
            },
        );
        store.put(
            "sk-team-missing",
            KeyStatus::Refused {
                expires_at: None,
                refusal: Refusal::TeamMissing {
                    team_id: "team-1".into(),
                },
            },
        );
        store.put(
            "sk-refused-expired",
            KeyStatus::Refused {
                expires_at: Some(SystemTime::now() - Duration::from_secs(1)),
                refusal: Refusal::Unsupported {
                    restriction: Restriction::RateLimits,
                    owner: Owner::Key,
                },
            },
        );
        Arc::new(store)
    }

    fn auth(store: &Arc<Store>) -> Auth {
        Auth::new(
            Arc::new(Master),
            Arc::new(LocalAdministrator),
            Arc::new(NoAdditionalPolicy),
            Arc::new(SystemClock),
        )
        .with_virtual_keys(Arc::new(CachedKeyLookup::new(store.clone(), settings())))
    }

    fn model(name: &str) -> AccessRequest {
        AccessRequest::Model {
            name: name.into(),
            deployment: format!("provider/{name}"),
            team_id: None,
        }
    }

    async fn call(auth: &Auth, token: &str, model_name: &str) -> Result<(), Error> {
        let identity = auth.authenticate(&SecretValue::new(token)).await?;
        identity.authorize(model(model_name)).await.map(|_| ())
    }

    #[rstest]
    #[case::master_key("sk-master", "anything", None)]
    #[case::key_with_all_models("sk-all", "anything", None)]
    #[case::listed_model("sk-scoped", "gpt-a", None)]
    #[case::wildcard_model("sk-scoped", "claude-opus-5-5-0", None)]
    #[case::unlisted_model("sk-scoped", "gpt-b", Some(403))]
    #[case::future_expiry("sk-later", "anything", None)]
    #[case::expired("sk-expired", "anything", Some(401))]
    #[case::blocked("sk-blocked", "anything", Some(401))]
    #[case::unknown("sk-unknown", "anything", Some(401))]
    #[case::team_allows("sk-team", "gpt-a", None)]
    #[case::team_narrows("sk-team", "gpt-b", Some(403))]
    #[case::over_budget("sk-over-budget", "anything", Some(422))]
    #[case::rate_limited("sk-rate-limited", "anything", Some(401))]
    #[case::team_blocked("sk-team-blocked", "anything", Some(401))]
    #[case::team_missing("sk-team-missing", "anything", Some(404))]
    #[tokio::test]
    async fn authorizes_keys_against_the_store(
        store: Arc<Store>,
        #[case] token: &str,
        #[case] model_name: &str,
        #[case] failure: Option<u16>,
    ) {
        let result = call(&auth(&store), token, model_name).await;
        assert_eq!(result.err().map(|error| error.status().as_u16()), failure);
    }

    #[rstest]
    #[case::team_key_shared("sk-team", None, true)]
    #[case::team_key_own_team("sk-team", Some("team-1"), true)]
    #[case::team_key_other_team("sk-team", Some("team-2"), false)]
    #[case::teamless_key_shared("sk-all", None, true)]
    #[case::teamless_key_team_deployment("sk-all", Some("team-1"), false)]
    #[case::master_key_team_deployment("sk-master", Some("team-2"), true)]
    #[tokio::test]
    async fn virtual_keys_only_reach_their_own_teams_deployments(
        store: Arc<Store>,
        #[case] token: &str,
        #[case] owner: Option<&str>,
        #[case] allowed: bool,
    ) {
        let identity = auth(&store)
            .authenticate(&SecretValue::new(token))
            .await
            .unwrap();
        let access = AccessRequest::Model {
            name: "gpt-a".into(),
            deployment: "provider/gpt-a".into(),
            team_id: owner.map(str::to_owned),
        };

        assert_eq!(identity.may_use_deployment(owner), allowed);
        let result = identity.authorize(access).await;
        assert_eq!(
            result.err().map(|error| error.status().as_u16()),
            (!allowed).then_some(403)
        );
    }

    #[rstest]
    #[tokio::test]
    async fn model_denial_and_blocked_keys_carry_the_proxy_messages(store: Arc<Store>) {
        let auth = auth(&store);
        assert!(matches!(
            call(&auth, "sk-scoped", "gpt-b").await,
            Err(Error::ModelAccessDenied { model, by: ModelDenial::Key }) if model == "gpt-b"
        ));
        assert!(matches!(
            call(&auth, "sk-team", "gpt-b").await,
            Err(Error::ModelAccessDenied { model, by: ModelDenial::Team }) if model == "gpt-b"
        ));
        assert!(matches!(
            call(&auth, "sk-team-blocked", "gpt-a").await,
            Err(Error::Refused(Refusal::TeamBlocked { team_id })) if team_id == "team-1"
        ));
        assert!(matches!(
            call(&auth, "sk-refused-expired", "gpt-a").await,
            Err(Error::Expired)
        ));
        assert!(matches!(
            call(&auth, "sk-blocked", "gpt-a").await,
            Err(Error::Blocked)
        ));
        assert!(matches!(
            call(&auth, "sk-expired", "gpt-a").await,
            Err(Error::Expired)
        ));
    }

    #[rstest]
    #[tokio::test]
    async fn virtual_keys_cannot_reach_mcp_or_ui(store: Arc<Store>) {
        let identity = auth(&store)
            .authenticate(&SecretValue::new("sk-all"))
            .await
            .unwrap();
        assert!(
            identity
                .authorize(AccessRequest::Route {
                    method: "POST".into(),
                    path: "/v1/messages".into(),
                })
                .await
                .is_ok()
        );
        assert!(matches!(
            identity
                .authorize(AccessRequest::Mcp {
                    server: "server".into(),
                    action: crate::McpAction::ListTools,
                })
                .await,
            Err(Error::Forbidden)
        ));
        assert_eq!(
            identity.caller().authentication().method,
            AuthenticationMethod::VirtualKey
        );
    }

    #[rstest]
    #[case::listed_route("sk-anthropic-only", "/v1/messages", None)]
    #[case::unlisted_route("sk-anthropic-only", "/v1/chat/completions", Some(403))]
    #[case::no_route_list("sk-all", "/v1/chat/completions", None)]
    #[case::master_key("sk-master", "/v1/chat/completions", None)]
    #[tokio::test]
    async fn checks_allowed_routes(
        store: Arc<Store>,
        #[case] token: &str,
        #[case] path: &str,
        #[case] failure: Option<u16>,
    ) {
        let identity = auth(&store)
            .authenticate(&SecretValue::new(token))
            .await
            .unwrap();
        let result = identity
            .authorize(AccessRequest::Route {
                method: "POST".into(),
                path: path.into(),
            })
            .await;
        assert_eq!(result.err().map(|error| error.status().as_u16()), failure);
    }

    #[rstest]
    #[tokio::test]
    async fn master_key_works_without_a_store(store: Arc<Store>) {
        store.down.store(true, Ordering::SeqCst);
        let auth = auth(&store);
        assert!(call(&auth, "sk-master", "gpt-a").await.is_ok());
        assert_eq!(store.reads.load(Ordering::SeqCst), 0);
        assert_eq!(
            call(&auth, "sk-all", "gpt-a")
                .await
                .unwrap_err()
                .status()
                .as_u16(),
            503
        );
    }

    #[rstest]
    #[tokio::test]
    async fn caches_found_keys_for_the_positive_ttl(store: Arc<Store>) {
        let auth = auth(&store);
        call(&auth, "sk-all", "gpt-a").await.unwrap();
        call(&auth, "sk-all", "gpt-a").await.unwrap();
        call(&auth, "sk-all", "gpt-a").await.unwrap();
        assert_eq!(store.reads.load(Ordering::SeqCst), 1);
        store.put("sk-all", KeyStatus::Blocked);
        assert!(call(&auth, "sk-all", "gpt-a").await.is_ok());
        tokio::time::sleep(Duration::from_millis(250)).await;
        assert!(matches!(
            call(&auth, "sk-all", "gpt-a").await,
            Err(Error::Blocked)
        ));
        assert_eq!(store.reads.load(Ordering::SeqCst), 2);
    }

    #[rstest]
    #[tokio::test]
    async fn caches_unknown_keys_for_the_negative_ttl(store: Arc<Store>) {
        let auth = auth(&store);
        assert!(call(&auth, "sk-new", "gpt-a").await.is_err());
        store.put("sk-new", active(&[], None));
        assert!(call(&auth, "sk-new", "gpt-a").await.is_err());
        assert_eq!(store.reads.load(Ordering::SeqCst), 1);
        tokio::time::sleep(Duration::from_millis(150)).await;
        assert!(call(&auth, "sk-new", "gpt-a").await.is_ok());
        assert_eq!(store.reads.load(Ordering::SeqCst), 2);
    }

    #[rstest]
    #[tokio::test]
    async fn concurrent_misses_share_one_store_read(store: Arc<Store>) {
        let auth = auth(&store);
        let calls = (0..16).map(|_| call(&auth, "sk-all", "gpt-a"));
        futures_util::future::join_all(calls)
            .await
            .into_iter()
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        assert_eq!(store.reads.load(Ordering::SeqCst), 1);
    }

    #[rstest]
    #[tokio::test]
    async fn known_keys_keep_working_while_the_store_is_down(store: Arc<Store>) {
        let auth = auth(&store);
        call(&auth, "sk-scoped", "gpt-a").await.unwrap();
        store.down.store(true, Ordering::SeqCst);
        tokio::time::sleep(Duration::from_millis(250)).await;

        assert!(call(&auth, "sk-scoped", "gpt-a").await.is_ok());
        assert!(matches!(
            call(&auth, "sk-scoped", "gpt-b").await,
            Err(Error::ModelAccessDenied { .. })
        ));
        let reads = store.reads.load(Ordering::SeqCst);
        assert_eq!(
            call(&auth, "sk-all", "gpt-a")
                .await
                .unwrap_err()
                .status()
                .as_u16(),
            503
        );
        // The failed read paused the store, so the never-seen key failed without another read.
        assert_eq!(store.reads.load(Ordering::SeqCst), reads);

        store.down.store(false, Ordering::SeqCst);
        tokio::time::sleep(Duration::from_millis(150)).await;
        assert!(call(&auth, "sk-all", "gpt-a").await.is_ok());
    }

    /// A store whose keys belong to `user-1` on `team-1`, counting reads per key.
    #[derive(Default)]
    struct Owned {
        reads: Mutex<HashMap<KeyHash, usize>>,
        blocked: AtomicBool,
    }

    impl KeyLookup for Owned {
        fn lookup<'a>(&'a self, _: &'a KeyHash) -> LookupFuture<'a> {
            unreachable!("the cache reads entries")
        }

        fn lookup_entry<'a>(&'a self, hash: &'a KeyHash) -> crate::keys::EntryFuture<'a> {
            Box::pin(async move {
                *self.reads.lock().unwrap().entry(hash.clone()).or_default() += 1;
                tokio::task::yield_now().await;
                Ok(KeyEntry {
                    status: Some(match self.blocked.load(Ordering::SeqCst) {
                        true => KeyStatus::Blocked,
                        false => active(&[], None),
                    }),
                    team_id: Some("team-1".into()),
                    user_id: Some("user-1".into()),
                })
            })
        }
    }

    fn hash(token: &str) -> KeyHash {
        KeyHash::from_token(&SecretValue::new(token))
    }

    #[rstest]
    #[case::key_hash(None, &[true, false])]
    #[case::team(Some("team_id:team-1"), &[true, true])]
    #[case::user(Some("user-1"), &[true, true])]
    #[case::membership(Some("team_membership:user-1:team-1"), &[true, true])]
    #[case::other_team(Some("team_id:team-2"), &[false, false])]
    #[case::other_user(Some("user-2"), &[false, false])]
    #[tokio::test]
    async fn invalidation_drops_the_keys_it_names(
        #[case] cache_key: Option<&str>,
        #[case] reread: &[bool],
    ) {
        let store = Arc::new(Owned::default());
        let keys = CachedKeyLookup::new(store.clone(), KeyCacheSettings::default());
        let tokens = ["sk-a", "sk-b"];
        for token in tokens {
            keys.lookup(&hash(token)).await.unwrap();
        }

        let first = hash("sk-a");
        keys.invalidate(cache_key.unwrap_or(first.as_str())).await;
        for token in tokens {
            keys.lookup(&hash(token)).await.unwrap();
        }

        let reads = store.reads.lock().unwrap();
        let expected: Vec<usize> = reread.iter().map(|again| 1 + usize::from(*again)).collect();
        assert_eq!(
            tokens.map(|token| reads[&hash(token)]).to_vec(),
            expected,
            "{cache_key:?}"
        );
    }

    #[rstest]
    #[tokio::test]
    async fn a_revoked_key_is_refused_as_soon_as_it_is_invalidated() {
        let store = Arc::new(Owned::default());
        let keys = CachedKeyLookup::new(store.clone(), KeyCacheSettings::default());
        let key = hash("sk-a");
        assert!(matches!(
            keys.lookup(&key).await,
            Ok(Some(KeyStatus::Active { .. }))
        ));

        store.blocked.store(true, Ordering::SeqCst);
        keys.invalidate(key.as_str()).await;

        assert_eq!(keys.lookup(&key).await.unwrap(), Some(KeyStatus::Blocked));
    }

    #[rstest]
    #[tokio::test]
    async fn invalidate_all_rereads_every_key() {
        let store = Arc::new(Owned::default());
        let keys = CachedKeyLookup::new(store.clone(), KeyCacheSettings::default());
        keys.lookup(&hash("sk-a")).await.unwrap();

        keys.invalidate_all();
        keys.lookup(&hash("sk-a")).await.unwrap();

        assert_eq!(store.reads.lock().unwrap()[&hash("sk-a")], 2);
    }

    #[rstest]
    #[tokio::test]
    async fn a_read_overlapping_an_invalidation_is_not_cached() {
        let store = Arc::new(Owned::default());
        let keys = Arc::new(CachedKeyLookup::new(
            store.clone(),
            KeyCacheSettings::default(),
        ));
        let key = hash("sk-a");
        let invalidating = {
            let keys = keys.clone();
            let key = key.clone();
            async move { keys.invalidate(key.as_str()).await }
        };
        // The invalidation lands while the read waits on the store.
        let (read, ()) = tokio::join!(keys.lookup(&key), invalidating);
        read.unwrap();
        keys.lookup(&key).await.unwrap();

        assert_eq!(store.reads.lock().unwrap()[&key], 2);
    }

    #[rstest]
    #[tokio::test]
    async fn slow_store_reads_time_out() {
        struct Hanging;
        impl KeyLookup for Hanging {
            fn lookup<'a>(&'a self, _: &'a KeyHash) -> LookupFuture<'a> {
                Box::pin(std::future::pending())
            }
        }
        let keys = CachedKeyLookup::new(
            Arc::new(Hanging),
            KeyCacheSettings {
                lookup_timeout: Duration::from_millis(20),
                ..settings()
            },
        );
        let hash = KeyHash::from_token(&SecretValue::new("sk-all"));
        assert!(matches!(keys.lookup(&hash).await, Err(KeyError::Lookup(_))));
    }

    #[rstest]
    #[tokio::test]
    async fn rejects_transport_credentials(store: Arc<Store>) {
        let authenticator = VirtualKeyAuthenticator::new(
            Arc::new(Master),
            Arc::new(CachedKeyLookup::new(store, settings())),
            Arc::new(SystemClock),
        );
        let parts = Request::new(()).into_parts().0;
        assert!(matches!(
            authenticator.verify(&Credential::Transport, &parts).await,
            Err(Error::InvalidToken)
        ));
    }
}
