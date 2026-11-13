use std::time::Duration;

use litellm_config::UnregisteredJwtClientBehavior as Behavior;
use serde_json::Value;

use super::{Claims, JwtSettings};
use crate::{
    AuthHost, Error, JwtDenial, KeyHash, Rows,
    authn::key::Keyed,
    host::KeyRow,
    identity::jwt::{
        JwtIdentity, JwtOutcome, UserLookup, UserRead,
        claims::{self, InternalClaim, nested_value},
    },
    value::id_text,
};

/// Cached for an identity that `auth_builder` resolved to a proxy admin under `auto_register`.
pub const PROXY_ADMIN_SENTINEL: &str = "__JWT_PROXY_ADMIN__";
pub const NO_MAPPING_SENTINEL: &str = "__NO_MAPPING__";

/// What the mapping cache holds for a claim. Anything that is not a sentinel is a key's hash.
#[derive(Clone, Debug, PartialEq, Eq)]
enum CachedMapping {
    ProxyAdmin,
    NoMapping,
    Key(KeyHash),
}

impl CachedMapping {
    fn read(cached: Option<&Value>) -> Option<Self> {
        Some(match cached?.as_str()? {
            PROXY_ADMIN_SENTINEL => Self::ProxyAdmin,
            NO_MAPPING_SENTINEL => Self::NoMapping,
            hash => Self::Key(KeyHash::new(hash)),
        })
    }

    fn value(&self) -> Value {
        Value::String(
            match self {
                Self::ProxyAdmin => PROXY_ADMIN_SENTINEL,
                Self::NoMapping => NO_MAPPING_SENTINEL,
                Self::Key(hash) => hash.as_str(),
            }
            .to_owned(),
        )
    }
}

/// The one cache key format for a claim's mapping. Python's management endpoints invalidate
/// through this too, so a key they evict is the key auth reads.
pub fn mapping_cache_key(claim_field: &str, claim_value: &str, issuer: Option<&str>) -> String {
    match issuer.filter(|issuer| !issuer.is_empty()) {
        None => format!("jwt_key_mapping:{claim_field}:{claim_value}"),
        Some(issuer) => format!("jwt_key_mapping:{issuer}:{claim_field}:{claim_value}"),
    }
}

/// `IdentityStore.resolve(hashed_token)` as the mapping reads it.
#[derive(Clone, Debug, PartialEq)]
pub struct MappedKey<K> {
    pub key: Keyed<K>,
    pub user_email: Option<String>,
}

/// The row `_auto_register_jwt_mapping` writes: a mapping onto `existing_hash`, or onto a key the
/// host mints with these ids when there is none.
#[derive(Clone, Copy, Debug)]
pub struct KeyMappingRow<'a> {
    pub claim_field: &'a str,
    pub claim_value: &'a str,
    pub issuer: Option<&'a str>,
    pub behavior: Behavior,
    pub existing_hash: Option<&'a KeyHash>,
    pub team_id: Option<&'a str>,
    pub user_id: Option<&'a str>,
    pub org_id: Option<&'a str>,
    pub agent_id: Option<&'a str>,
}

/// How the mapping write ended. On a unique conflict the host deletes the key it minted and
/// re-reads the winner's mapping, which may have been deleted in between.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum KeyMappingWrite {
    Created(KeyHash),
    Conflict(KeyHash),
    ConflictVanished,
}

#[derive(Clone, Debug, PartialEq)]
pub enum KeyOrigin {
    Mapping,
    /// `minted` keys take the identity's `org_id`.
    AutoRegistered {
        minted: bool,
        org_id: Option<String>,
    },
}

/// What a JWT mapped onto a virtual key keeps from its claims once the key takes over, and the
/// user the policy loaded for it before an auto-registration.
#[derive(Clone, Debug, PartialEq)]
pub struct MappedJwt<U> {
    pub claims: Value,
    pub end_user_id: Option<String>,
    pub user_email: Option<String>,
    pub origin: KeyOrigin,
    pub user: Option<U>,
}

/// `_PendingAutoRegister`
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Pending {
    pub(crate) claim_field: String,
    pub(crate) claim_value: String,
    pub(crate) cache_key: String,
    pub(crate) issuer: Option<String>,
    pub(crate) behavior: Behavior,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) enum Resolved<K> {
    Key(MappedKey<K>),
    Pending(Pending),
    Unmapped,
}

fn ttl(settings: &JwtSettings) -> Duration {
    Duration::from_secs_f64(settings.config.virtual_key_mapping_cache_ttl.max(0.0))
}

async fn key<H: AuthHost>(host: &H, hash: &KeyHash) -> Result<Resolved<KeyRow<H>>, Error> {
    Ok(host
        .key_by_hash(hash)
        .await?
        .map_or(Resolved::Unmapped, Resolved::Key))
}

async fn cache(
    host: &impl AuthHost,
    key: &str,
    mapping: &CachedMapping,
    settings: &JwtSettings,
) -> Result<(), Error> {
    host.cache_json(key, &mapping.value(), ttl(settings)).await
}

/// `_lookup_jwt_mapping_token_hash`: the issuer's row, else the global row, each cached under its
/// own scope's key.
async fn lookup_hash(
    host: &impl AuthHost,
    pending: &Pending,
    settings: &JwtSettings,
) -> Result<Option<KeyHash>, Error> {
    let (field, value, issuer) = (
        &pending.claim_field,
        &pending.claim_value,
        pending.issuer.as_deref(),
    );
    if let Some(hash) = host.key_mapping_hash(field, value, issuer).await? {
        let mapping = CachedMapping::Key(hash.clone());
        cache(host, &pending.cache_key, &mapping, settings).await?;
        return Ok(Some(hash));
    }
    if issuer.is_none() {
        return Ok(None);
    }
    let global_key = mapping_cache_key(field, value, None);
    match CachedMapping::read(host.cached_json(&global_key).await?.as_ref()) {
        Some(CachedMapping::Key(hash)) => return Ok(Some(hash)),
        // `_lookup_jwt_mapping_token_hash` reads the proxy admin sentinel under the global key as a
        // hash, which no key has.
        Some(CachedMapping::ProxyAdmin) => return Ok(Some(KeyHash::new(PROXY_ADMIN_SENTINEL))),
        Some(CachedMapping::NoMapping) | None => {}
    }
    let global = host.key_mapping_hash(field, value, None).await?;
    if let Some(hash) = &global {
        let mapping = CachedMapping::Key(hash.clone());
        cache(host, &global_key, &mapping, settings).await?;
    }
    Ok(global)
}

/// What `_resolve_jwt_to_virtual_key` does with a claim no key is mapped to.
enum Unregistered {
    Reject,
    NeedsDb,
    Pending,
    Fallback,
}

fn unregistered(behavior: Behavior, has_db: bool) -> Unregistered {
    match behavior {
        Behavior::Reject => Unregistered::Reject,
        Behavior::AutoRegister | Behavior::LinkOnly if !has_db => Unregistered::NeedsDb,
        Behavior::AutoRegister | Behavior::LinkOnly => Unregistered::Pending,
        Behavior::FallbackTeamMapping => Unregistered::Fallback,
    }
}

/// `_resolve_jwt_to_virtual_key`
pub(crate) async fn resolve<H: AuthHost>(
    host: &H,
    claims: &Claims,
    settings: &JwtSettings,
) -> Result<Resolved<KeyRow<H>>, Error> {
    let issuer = claims
        .0
        .get(<&str>::from(InternalClaim::Issuer))
        .and_then(Value::as_str);
    let Some(claim_field) = settings.config.virtual_key_claim_field(issuer) else {
        return Ok(Resolved::Unmapped);
    };
    let behavior = settings.config.unregistered_jwt_client_behavior(issuer);
    let Some(claim_value) = nested_value(&claims.0, claim_field).and_then(id_text) else {
        return match behavior {
            Behavior::FallbackTeamMapping => Ok(Resolved::Unmapped),
            Behavior::Reject | Behavior::AutoRegister | Behavior::LinkOnly => {
                Err(JwtDenial::MappingClaimMissing {
                    claim_field: claim_field.to_string(),
                }
                .into())
            }
        };
    };
    let pending = Pending {
        cache_key: mapping_cache_key(claim_field.as_str(), &claim_value, issuer),
        claim_field: claim_field.to_string(),
        claim_value,
        issuer: issuer.map(str::to_owned),
        behavior,
    };
    let no_mapping = || JwtDenial::NoMapping {
        claim_field: pending.claim_field.clone(),
        claim_value: pending.claim_value.clone(),
    };
    let needs_db = JwtDenial::MappingNeedsDb { behavior };

    let cached = host.cached_json(&pending.cache_key).await?;
    match CachedMapping::read(cached.as_ref()) {
        Some(CachedMapping::ProxyAdmin) if behavior == Behavior::AutoRegister => {
            return Ok(Resolved::Unmapped);
        }
        Some(CachedMapping::ProxyAdmin) | None => {}
        Some(CachedMapping::NoMapping) => {
            return match unregistered(behavior, settings.has_db) {
                Unregistered::Reject => Err(no_mapping().into()),
                Unregistered::NeedsDb => Err(needs_db.into()),
                Unregistered::Pending => {
                    host.forget_json(&pending.cache_key).await?;
                    Ok(Resolved::Pending(pending))
                }
                Unregistered::Fallback => Ok(Resolved::Unmapped),
            };
        }
        Some(CachedMapping::Key(hash)) => return key(host, &hash).await,
    }

    let hash = match settings.has_db {
        true => lookup_hash(host, &pending, settings).await?,
        false => None,
    };
    if let Some(hash) = hash {
        return key(host, &hash).await;
    }
    match unregistered(behavior, settings.has_db) {
        Unregistered::Reject => {
            cache(
                host,
                &pending.cache_key,
                &CachedMapping::NoMapping,
                settings,
            )
            .await?;
            Err(no_mapping().into())
        }
        Unregistered::NeedsDb => Err(needs_db.into()),
        Unregistered::Pending => Ok(Resolved::Pending(pending)),
        Unregistered::Fallback => {
            cache(
                host,
                &pending.cache_key,
                &CachedMapping::NoMapping,
                settings,
            )
            .await?;
            Ok(Resolved::Unmapped)
        }
    }
}

/// The mapping `_auto_register_jwt_mapping` writes for the identity `auth_builder` authorized:
/// onto the user's reusable key, or onto a key the host mints. Link-only needs a key to reuse.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Registration {
    pending: Pending,
    existing: Option<KeyHash>,
    team_id: Option<String>,
    user_id: Option<String>,
    org_id: Option<String>,
    agent_id: Option<String>,
}

impl Registration {
    pub(crate) fn row(&self) -> KeyMappingRow<'_> {
        KeyMappingRow {
            claim_field: &self.pending.claim_field,
            claim_value: &self.pending.claim_value,
            issuer: self.pending.issuer.as_deref(),
            behavior: self.pending.behavior,
            existing_hash: self.existing.as_ref(),
            team_id: self.team_id.as_deref(),
            user_id: self.user_id.as_deref(),
            org_id: self.org_id.as_deref(),
            agent_id: self.agent_id.as_deref(),
        }
    }
}

pub(crate) async fn registration<H: AuthHost>(
    host: &H,
    pending: &Pending,
    identity: &JwtIdentity<H::Rows>,
    settings: &JwtSettings,
) -> Result<Registration, Error> {
    let link_only = pending.behavior == Behavior::LinkOnly;
    let reuse_existing = link_only || settings.config.auto_register_map_existing_key;
    let existing = match identity.user_id.as_deref() {
        Some(user_id)
            if reuse_existing
                && settings
                    .config
                    .is_user_identity_claim(&pending.claim_field, pending.issuer.as_deref()) =>
        {
            host.reusable_key_hash(user_id, identity.team_id.as_deref())
                .await?
        }
        _ => None,
    };
    if link_only && existing.is_none() {
        return Err(JwtDenial::NoEligibleKey.into());
    }
    Ok(Registration {
        pending: pending.clone(),
        existing,
        team_id: identity.team_id.clone(),
        user_id: identity.user_id.clone(),
        org_id: identity.org_id.clone(),
        agent_id: identity.agent_id.clone(),
    })
}

/// The key a written registration maps to, cached under its claim, and whether it was minted.
/// `None` when the mapped key does not load.
pub(crate) async fn registered<H: AuthHost>(
    host: &H,
    registration: &Registration,
    write: KeyMappingWrite,
    settings: &JwtSettings,
) -> Result<Option<(MappedKey<KeyRow<H>>, bool)>, Error> {
    let hash = match write {
        KeyMappingWrite::Created(hash) | KeyMappingWrite::Conflict(hash) => hash,
        KeyMappingWrite::ConflictVanished => {
            return Err(JwtDenial::MappingRaceVanished {
                behavior: registration.pending.behavior,
            }
            .into());
        }
    };
    let mapping = CachedMapping::Key(hash.clone());
    cache(host, &registration.pending.cache_key, &mapping, settings).await?;
    Ok(host
        .key_by_hash(&hash)
        .await?
        .map(|key| (key, registration.existing.is_none())))
}

/// Caches that the claim belongs to a proxy admin, who needs no mapped key.
pub(crate) async fn remember_proxy_admin(
    host: &impl AuthHost,
    pending: &Pending,
    settings: &JwtSettings,
) -> Result<(), Error> {
    cache(
        host,
        &pending.cache_key,
        &CachedMapping::ProxyAdmin,
        settings,
    )
    .await
}

/// The mapped key's `end_user_id` and the owner's email it backfills from the claims.
async fn mapped_outcome<H: AuthHost>(
    host: &H,
    claims: &Claims,
    mapped: MappedKey<KeyRow<H>>,
    settings: &JwtSettings,
) -> JwtOutcome<H::Rows> {
    let user_email = match (mapped.key.record.user_id.as_deref(), mapped.user_email) {
        (Some(owner), None) => match (
            claims::user_email(claims, settings),
            claims::user_id(claims, settings, None),
        ) {
            (Some(email), Some(jwt_user_id)) if jwt_user_id == owner => {
                let lookup = UserLookup {
                    user_id: owner,
                    user_email: Some(&email),
                    sso_user_id: None,
                    upsert: false,
                };
                let user = host.user(&lookup).await.ok().and_then(UserRead::found);
                user.and_then(|user| user.user_email)
            }
            _ => None,
        },
        (_, user_email) => user_email,
    };
    let jwt = MappedJwt {
        claims: Value::Object(claims.0.clone()),
        end_user_id: claims::end_user_id(claims, settings).filter(|text| !text.is_empty()),
        user_email,
        origin: KeyOrigin::Mapping,
        user: None,
    };
    JwtOutcome::MappedKey(mapped.key, jwt)
}

pub(crate) enum Mapping<R: Rows> {
    Mapped(Box<JwtOutcome<R>>),
    Pending(Pending),
    Unmapped,
}

/// The virtual key a verified JWT maps to, before anything else is read about its caller. A
/// managed agent has to authenticate with its JWT directly.
pub(crate) async fn mapped_key<H: AuthHost>(
    host: &H,
    claims: &Claims,
    settings: &JwtSettings,
) -> Result<Mapping<H::Rows>, Error> {
    if !settings.config.is_virtual_key_mapping_configured() {
        return Ok(Mapping::Unmapped);
    }
    if !claims.0.is_empty()
        && host
            .managed_agent(&Value::Object(claims.0.clone()))
            .await?
            .is_some()
    {
        return Err(JwtDenial::ManagedAgentNeedsDirectJwt.into());
    }
    Ok(match resolve(host, claims, settings).await? {
        Resolved::Key(mapped) => Mapping::Mapped(Box::new(
            mapped_outcome(host, claims, mapped, settings).await,
        )),
        Resolved::Pending(pending) => Mapping::Pending(pending),
        Resolved::Unmapped => Mapping::Unmapped,
    })
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use litellm_config::{JwtAuthSettings, JwtIssuerConfig};
    use rstest::rstest;
    use serde_json::json;

    use super::*;
    use crate::{
        authn::key::KeyRecord,
        commit::{self, Grant, Grants},
        identity::jwt::JwtRows,
        testing::{Labels, ScriptedHost},
    };

    /// Registration, its write through `commit`, then the key it left mapped.
    async fn auto_register(
        host: &ScriptedHost,
        pending: &Pending,
        identity: &JwtIdentity<Labels>,
        settings: &JwtSettings,
    ) -> Result<Option<(MappedKey<String>, bool)>, Error> {
        let registration = registration(host, pending, identity, settings).await?;
        let grants = Grants::default().with(Grant::CreateKeyMapping(registration.clone()));
        let applied = commit::apply(host, grants).await?;
        match applied.key_mapping {
            Some(write) => registered(host, &registration, write, settings).await,
            None => Ok(None),
        }
    }

    const FIELD: &str = "client_id";

    fn identity() -> JwtIdentity<Labels> {
        JwtIdentity {
            claims: json!({}),
            is_proxy_admin: false,
            user_id: None,
            user_email: None,
            team_id: None,
            org_id: None,
            end_user_id: None,
            agent_id: None,
            managed_agent: None,
            rows: JwtRows::default(),
        }
    }

    fn settings(behavior: Behavior, has_db: bool) -> JwtSettings {
        JwtSettings {
            config: JwtAuthSettings {
                virtual_key_claim_field: Some(FIELD.into()),
                unregistered_jwt_client_behavior: behavior,
                user_id_jwt_field: Some("sub".into()),
                ..JwtAuthSettings::default()
            },
            has_db,
            ..JwtSettings::default()
        }
    }

    fn claims(value: Value) -> Claims {
        Claims(value.as_object().unwrap().clone())
    }

    fn mapped(hash: &str) -> MappedKey<String> {
        MappedKey {
            key: Keyed {
                record: KeyRecord {
                    token: Some(KeyHash::new(hash)),
                    ..KeyRecord::default()
                },
                row: hash.to_owned(),
            },
            user_email: None,
        }
    }

    fn pending(behavior: Behavior) -> Resolved<String> {
        Resolved::Pending(Pending {
            claim_field: FIELD.to_owned(),
            claim_value: "app".to_owned(),
            cache_key: "jwt_key_mapping:client_id:app".to_owned(),
            issuer: None,
            behavior,
        })
    }

    #[rstest]
    #[case::global(None, "jwt_key_mapping:f:v")]
    #[case::empty_issuer_is_global(Some(""), "jwt_key_mapping:f:v")]
    #[case::issuer_scoped(Some("https://idp"), "jwt_key_mapping:https://idp:f:v")]
    fn cache_key_is_scoped_by_issuer(#[case] issuer: Option<&str>, #[case] expected: &str) {
        assert_eq!(mapping_cache_key("f", "v", issuer), expected);
    }

    #[rstest]
    #[case::fallback_has_nothing_to_map(Behavior::FallbackTeamMapping, Ok(Resolved::Unmapped))]
    #[case::reject_denies(Behavior::Reject, Err(JwtDenial::MappingClaimMissing { claim_field: FIELD.to_owned() }.into()))]
    #[case::auto_register_denies(
        Behavior::AutoRegister,
        Err(JwtDenial::MappingClaimMissing { claim_field: FIELD.to_owned() }.into())
    )]
    #[tokio::test]
    async fn missing_claim_applies_the_no_match_policy(
        #[case] behavior: Behavior,
        #[case] expected: Result<Resolved<String>, Error>,
        #[values(json!(null), json!(true), json!(1.5), json!({"id": "app"}))] claim: Value,
    ) {
        let host = ScriptedHost::default();
        let resolved = resolve(
            &host,
            &claims(json!({"client_id": claim})),
            &settings(behavior, true),
        )
        .await;
        assert_eq!(resolved, expected);
        assert!(host.calls().is_empty());
    }

    #[rstest]
    #[case::hash_loads_the_key(
        Behavior::Reject,
        json!("h1"),
        Ok(Resolved::Key(mapped("h1"))),
        &["get jwt_key_mapping:client_id:app", "key_by_hash h1"]
    )]
    #[case::admin_sentinel_skips_lookup_under_auto_register(
        Behavior::AutoRegister,
        json!(PROXY_ADMIN_SENTINEL),
        Ok(Resolved::Unmapped),
        &["get jwt_key_mapping:client_id:app"]
    )]
    #[case::admin_sentinel_is_ignored_by_other_policies(
        Behavior::FallbackTeamMapping,
        json!(PROXY_ADMIN_SENTINEL),
        Ok(Resolved::Unmapped),
        &[
            "get jwt_key_mapping:client_id:app",
            "mapping client_id=app -",
            "set jwt_key_mapping:client_id:app 300",
        ]
    )]
    #[case::no_mapping_rejects(
        Behavior::Reject,
        json!(NO_MAPPING_SENTINEL),
        Err(JwtDenial::NoMapping { claim_field: FIELD.to_owned(), claim_value: "app".to_owned() }.into()),
        &["get jwt_key_mapping:client_id:app"]
    )]
    #[case::stale_no_mapping_is_evicted_for_auto_register(
        Behavior::AutoRegister,
        json!(NO_MAPPING_SENTINEL),
        Ok(pending(Behavior::AutoRegister)),
        &["get jwt_key_mapping:client_id:app", "forget jwt_key_mapping:client_id:app"]
    )]
    #[case::no_mapping_falls_back(
        Behavior::FallbackTeamMapping,
        json!(NO_MAPPING_SENTINEL),
        Ok(Resolved::Unmapped),
        &["get jwt_key_mapping:client_id:app"]
    )]
    #[tokio::test]
    async fn cached_mapping(
        #[case] behavior: Behavior,
        #[case] cached: Value,
        #[case] expected: Result<Resolved<String>, Error>,
        #[case] calls: &[&str],
    ) {
        let host = ScriptedHost {
            json_cache: HashMap::from([("jwt_key_mapping:client_id:app".to_owned(), cached)])
                .into(),
            keys: HashMap::from([("h1", mapped("h1"))]),
            ..ScriptedHost::default()
        };
        let resolved = resolve(
            &host,
            &claims(json!({"client_id": "app"})),
            &settings(behavior, true),
        )
        .await;
        assert_eq!(resolved, expected);
        assert_eq!(host.calls(), calls);
    }

    #[rstest]
    #[case::reject_caches_the_miss(
        Behavior::Reject,
        true,
        Err(JwtDenial::NoMapping { claim_field: FIELD.to_owned(), claim_value: "app".to_owned() }.into()),
        &["get jwt_key_mapping:client_id:app", "mapping client_id=app -", "set jwt_key_mapping:client_id:app 300"]
    )]
    #[case::auto_register_defers(
        Behavior::AutoRegister,
        true,
        Ok(pending(Behavior::AutoRegister)),
        &["get jwt_key_mapping:client_id:app", "mapping client_id=app -"]
    )]
    #[case::link_only_without_db(
        Behavior::LinkOnly,
        false,
        Err(JwtDenial::MappingNeedsDb { behavior: Behavior::LinkOnly }.into()),
        &["get jwt_key_mapping:client_id:app"]
    )]
    #[case::fallback_caches_the_miss(
        Behavior::FallbackTeamMapping,
        true,
        Ok(Resolved::Unmapped),
        &["get jwt_key_mapping:client_id:app", "mapping client_id=app -", "set jwt_key_mapping:client_id:app 300"]
    )]
    #[tokio::test]
    async fn uncached_miss(
        #[case] behavior: Behavior,
        #[case] has_db: bool,
        #[case] expected: Result<Resolved<String>, Error>,
        #[case] calls: &[&str],
    ) {
        let host = ScriptedHost::default();
        let resolved = resolve(
            &host,
            &claims(json!({"client_id": "app"})),
            &settings(behavior, has_db),
        )
        .await;
        assert_eq!(resolved, expected);
        assert_eq!(host.calls(), calls);
        let cached = host
            .json_cache
            .lock()
            .unwrap()
            .get("jwt_key_mapping:client_id:app")
            .cloned();
        let caches_miss = calls.iter().any(|call| call.starts_with("set "));
        assert_eq!(cached, caches_miss.then(|| json!(NO_MAPPING_SENTINEL)));
    }

    #[tokio::test]
    async fn issuer_scope_falls_back_to_the_global_row_cached_under_its_own_key() {
        let host = ScriptedHost {
            mappings: HashMap::from([("client_id=7 -".to_owned(), "global".to_owned())]),
            keys: HashMap::from([("global", mapped("global"))]),
            ..ScriptedHost::default()
        };
        let settings = JwtSettings {
            config: JwtAuthSettings {
                issuers: Some(Box::new([JwtIssuerConfig {
                    issuer: "idp".to_owned(),
                    virtual_key_claim_field: Some(FIELD.into()),
                    unregistered_jwt_client_behavior: Some(Behavior::Reject),
                    ..JwtIssuerConfig::default()
                }])),
                ..JwtAuthSettings::default()
            },
            has_db: true,
            ..JwtSettings::default()
        };
        let claims = claims(json!({"client_id": 7, "_litellm_jwt_issuer": "idp"}));
        let resolved = resolve(&host, &claims, &settings).await;
        assert_eq!(resolved, Ok(Resolved::Key(mapped("global"))));
        assert_eq!(
            host.calls(),
            [
                "get jwt_key_mapping:idp:client_id:7",
                "mapping client_id=7 idp",
                "get jwt_key_mapping:client_id:7",
                "mapping client_id=7 -",
                "set jwt_key_mapping:client_id:7 300",
                "key_by_hash global",
            ]
        );
    }

    fn registering(behavior: Behavior) -> Pending {
        let Resolved::Pending(pending) = pending(behavior) else {
            unreachable!()
        };
        pending
    }

    #[rstest]
    #[case::mints_a_key(
        Behavior::AutoRegister,
        false,
        Ok(Some((mapped("minted"), true))),
        &["create_mapping client_id=app existing=-", "set jwt_key_mapping:client_id:app 300", "key_by_hash minted"]
    )]
    #[case::link_only_without_reusable_key(Behavior::LinkOnly, false, Err(JwtDenial::NoEligibleKey.into()), &["reusable u1 t1"])]
    #[case::link_only_reuses_the_users_key(
        Behavior::LinkOnly,
        true,
        Ok(Some((mapped("reused"), false))),
        &[
            "reusable u1 t1",
            "create_mapping client_id=app existing=reused",
            "set jwt_key_mapping:client_id:app 300",
            "key_by_hash reused",
        ]
    )]
    #[tokio::test]
    async fn auto_register_maps_the_authorized_identity(
        #[case] behavior: Behavior,
        #[case] reusable: bool,
        #[case] expected: Result<Option<(MappedKey<String>, bool)>, Error>,
        #[case] calls: &[&str],
    ) {
        let host = ScriptedHost {
            reusable_key: reusable.then_some("reused"),
            keys: HashMap::from([("minted", mapped("minted")), ("reused", mapped("reused"))]),
            ..ScriptedHost::default()
        };
        let settings = JwtSettings {
            config: JwtAuthSettings {
                user_id_jwt_field: Some(FIELD.into()),
                ..settings(behavior, true).config
            },
            ..settings(behavior, true)
        };
        let identity = JwtIdentity {
            user_id: Some("u1".to_owned()),
            team_id: Some("t1".to_owned()),
            ..identity()
        };
        let registered = auto_register(&host, &registering(behavior), &identity, &settings).await;
        assert_eq!(registered, expected);
        assert_eq!(host.calls(), calls);
    }

    #[tokio::test]
    async fn vanished_race_winner_is_a_503() {
        let host = ScriptedHost {
            key_mapping_write: Some(KeyMappingWrite::ConflictVanished),
            ..ScriptedHost::default()
        };
        let registered = auto_register(
            &host,
            &registering(Behavior::AutoRegister),
            &identity(),
            &settings(Behavior::AutoRegister, true),
        )
        .await;
        let denial = registered.unwrap_err();
        assert_eq!(
            denial,
            JwtDenial::MappingRaceVanished {
                behavior: Behavior::AutoRegister
            }
            .into()
        );
        assert_eq!(
            denial.public(),
            Some(crate::PublicDenial::Http {
                status: 503,
                detail: "JWT Key Mapping: AUTO_REGISTER race resolution failed — winner's mapping was concurrently \
                         removed. Retry the request."
                    .to_owned()
            })
        );
    }
}
