//! A virtual key as the auth cache holds it, as far as the native host reads it. Each field is taken
//! apart by name, so a field added to the cached key does not compile here until it is honored,
//! left to a host step or named as one no step is asked about.

use std::{collections::BTreeMap, sync::Arc};

use jiff::{Timestamp, civil::DateTime, tz::TimeZone};
use litellm_gateway_auth_core::{
    KeyHash, StoredKey,
    authn::key::{KeyRecord, Keyed},
};
use litellm_python_compat::{Value, datetime::IsoDateTime, json};
use litellm_redis_store::auth_cache::{Bag, KeyEntry, Lists, ObjectPermissionEntry};
use serde::Deserialize;

use crate::callbacks::parallel_request_limiter::{KeyLimits, ModelLimits};

/// The name of the first of `fields` that holds something.
macro_rules! carried {
    ($($field:ident),+ $(,)?) => {
        [$((stringify!($field), Carried::is_carried($field)),)+]
            .into_iter()
            .find(|(_, carried)| *carried)
            .map(|(field, _)| field)
    };
}

/// The key type the proxy gives a key generated without one, which reaches every route.
const DEFAULT_KEY_TYPE: &str = "default";

/// The `models` entries the proxy reads through a team or a user, which a key with neither does
/// not have.
const OWNER_MODELS: [&str; 2] = ["all-team-models", "no-default-models"];

/// Whether the auth core's model rule reads a `models` entry as the proxy does: a name, an
/// access group or `*`. A pattern like `anthropic/*` the proxy matches against providers is not read.
fn is_read(model: &str) -> bool {
    !OWNER_MODELS.contains(&model) && (model == "*" || !model.contains('*'))
}

pub(super) trait Carried {
    fn is_carried(&self) -> bool;
}

impl Carried for bool {
    fn is_carried(&self) -> bool {
        *self
    }
}

/// A value the key holds, whatever it is: a limit of zero and an empty owner are still set.
macro_rules! carried_when_held {
    ($($held:ty),+) => {
        $(impl Carried for $held {
            fn is_carried(&self) -> bool {
                true
            }
        })+
    };
}

carried_when_held!(String, i64, f64, IsoDateTime, ObjectPermissionEntry);

impl<T> Carried for Vec<T> {
    fn is_carried(&self) -> bool {
        !self.is_empty()
    }
}

impl Carried for Bag {
    fn is_carried(&self) -> bool {
        !self.is_empty()
    }
}

impl Carried for Lists {
    fn is_carried(&self) -> bool {
        !self.0.is_empty()
    }
}

impl Carried for Value {
    fn is_carried(&self) -> bool {
        !matches!(self, Self::None)
    }
}

impl<T: Carried> Carried for Option<T> {
    fn is_carried(&self) -> bool {
        self.as_ref().is_some_and(Carried::is_carried)
    }
}

/// The team the core reads no team row for, so no step declines a key in it.
const UI_TEAM: &str = "litellm-dashboard";

/// A virtual key the native host read, with what the host's own steps read of it.
#[derive(Clone, Debug, PartialEq)]
pub struct Key {
    pub(super) record: KeyRecord,
    /// The key's `models` list, which the auth core decides model access on.
    models: Arc<[String]>,
    /// The key's own rate limits, which the limiter enforces.
    limits: KeyLimits,
    /// The priority class the key's `metadata.priority` names, which reserves it a share of a
    /// model group's capacity.
    priority: Option<Arc<str>>,
    /// The key's `key_alias`, which its calls are logged under.
    alias: Option<Arc<str>>,
    /// The first budget column the row holds, which `key_budgets` declines.
    pub(super) budget: Option<&'static str>,
    /// The row maps model names, which `normalize_model` declines.
    pub(super) aliased: bool,
}

impl StoredKey for Key {
    fn record(&self) -> &KeyRecord {
        &self.record
    }
}

impl Key {
    pub(crate) fn models(&self) -> &[String] {
        &self.models
    }

    pub(crate) fn limits(&self) -> &KeyLimits {
        &self.limits
    }

    pub(crate) fn priority(&self) -> Option<&Arc<str>> {
        self.priority.as_ref()
    }

    pub(crate) fn alias(&self) -> Option<&str> {
        self.alias.as_deref()
    }
}

/// The `metadata` the native app honors: the key's own limit for each model it names one for,
/// which `/key/generate` stores from `model_rpm_limit` and `model_tpm_limit`, and the priority
/// class `dynamic_rate_limiter_v3` counts the key under, when it is written as text.
#[derive(Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct Metadata {
    #[serde(default)]
    model_rpm_limit: BTreeMap<String, i64>,
    #[serde(default)]
    model_tpm_limit: BTreeMap<String, i64>,
    priority: Option<String>,
}

impl Metadata {
    /// `None` when `metadata` holds anything else, or a limit that is not a whole number.
    fn read(metadata: &Bag) -> Option<Self> {
        if metadata.is_empty() {
            return Some(Self::default());
        }
        let text = json::dumps(&Value::Dict(metadata.entries().to_vec())).ok()?;
        serde_json::from_str(&text).ok()
    }
}

/// What the key holds beside the fields the core's record carries.
struct Held {
    /// The first column that holds something no host step is asked about.
    unasked: Option<&'static str>,
    budget: Option<&'static str>,
    aliased: bool,
}

/// The native app honors a key's expiry, its block, its model list, its `rpm_limit`, `tpm_limit`
/// and `max_parallel_requests`, and the per-model limits in its metadata. Its `tpd_limit` only
/// counts a batch, which no native route serves. The owner, the team and the project are the
/// core's to ask the host for, the budgets and the aliases are steps the host declines, and every
/// other field that holds something is one no step is asked about. The team, organization and
/// project fields are only filled for a key that names one, and the last group
/// only describes the key or the request that cached it. A field this build does not name, which
/// a newer pod wrote, is not asked about either unless that pod wrote it empty.
fn held(key: &KeyEntry) -> Held {
    let KeyEntry {
        token: _,
        key_name: _,
        key_alias: _,
        spend: _,
        total_spend: _,
        model_spend: _,
        expires: _,
        blocked: _,
        created_at: _,
        created_by: _,
        updated_at: _,
        updated_by: _,
        settings_updated_at: _,
        last_active: _,
        rotation_count: _,
        auto_rotate: _,
        rotation_interval: _,
        last_rotation_at: _,
        key_rotation_at: _,
        team_spend: _,
        team_alias: _,
        team_tpm_limit: _,
        team_rpm_limit: _,
        team_tpd_limit: _,
        team_max_budget: _,
        team_soft_budget: _,
        team_model_max_budget: _,
        team_models: _,
        team_blocked: _,
        team_model_aliases: _,
        team_member: _,
        team_metadata: _,
        team_object_permission_id: _,
        team_member_spend: _,
        team_member_tpm_limit: _,
        team_member_rpm_limit: _,
        organization_alias: _,
        organization_max_budget: _,
        organization_tpm_limit: _,
        organization_rpm_limit: _,
        organization_metadata: _,
        project_alias: _,
        max_parallel_requests: _,
        rpm_limit: _,
        tpm_limit: _,
        tpd_limit: _,
        metadata: _,
        key_type,
        soft_budget_cooldown,
        models,
        aliases,
        config,
        router_settings,
        user_id: _,
        team_id,
        agent_id,
        project_id: _,
        org_id,
        permissions,
        max_budget,
        soft_budget,
        budget_duration,
        budget_reset_at,
        budget_limits,
        budget_fallbacks,
        budget_id,
        model_max_budget,
        allowed_cache_controls,
        allowed_routes,
        access_group_ids,
        object_permission_id,
        allowed_model_region,
        customer_hierarchy,
        end_user_id,
        end_user_max_budget,
        end_user_model_max_budget,
        end_user_model_rpm_limit,
        end_user_model_tpm_limit,
        end_user_object_permission,
        end_user_rpm_limit,
        end_user_tpd_limit,
        end_user_tpm_limit,
        is_session_token,
        jwt_claims,
        litellm_budget_table,
        object_permission,
        project_metadata,
        rpm_limit_per_model,
        team_object_permission,
        tpm_limit_per_model,
        user_max_budget,
        user_model_max_budget,
        user_role,
        user_rpm_limit,
        user_tpm_limit,
        unread,
        api_key: _,
        created_by_user: _,
        key_id: _,
        last_refreshed_at: _,
        parent_otel_span: _,
        request_route: _,
        user: _,
        user_email: _,
        user_spend: _,
    } = key;
    let other_key_type = key_type
        .as_deref()
        .is_some_and(|key_type| key_type != DEFAULT_KEY_TYPE);
    let unread_models = models.iter().any(|model| !is_read(model));
    let ui_team = team_id.as_deref() == Some(UI_TEAM);
    let unread = &unread.iter().any(|(_, value)| value.is_carried());
    Held {
        unasked: other_key_type
            .then_some("key_type")
            .or_else(|| unread_models.then_some("models"))
            .or_else(|| ui_team.then_some("team_id"))
            .or_else(|| {
                carried![
                    config,
                    router_settings,
                    agent_id,
                    org_id,
                    permissions,
                    allowed_cache_controls,
                    allowed_routes,
                    access_group_ids,
                    object_permission_id,
                    allowed_model_region,
                    customer_hierarchy,
                    end_user_id,
                    end_user_max_budget,
                    end_user_model_max_budget,
                    end_user_model_rpm_limit,
                    end_user_model_tpm_limit,
                    end_user_object_permission,
                    end_user_rpm_limit,
                    end_user_tpd_limit,
                    end_user_tpm_limit,
                    is_session_token,
                    jwt_claims,
                    litellm_budget_table,
                    object_permission,
                    project_metadata,
                    rpm_limit_per_model,
                    team_object_permission,
                    tpm_limit_per_model,
                    user_max_budget,
                    user_model_max_budget,
                    user_role,
                    user_rpm_limit,
                    user_tpm_limit,
                    unread,
                ]
            }),
        budget: carried![
            soft_budget_cooldown,
            max_budget,
            soft_budget,
            budget_duration,
            budget_reset_at,
            budget_limits,
            budget_fallbacks,
            budget_id,
            model_max_budget,
        ],
        aliased: aliases.is_carried(),
    }
}

/// When the key stops answering. Python writes the time with an offset, and reads one written
/// without as UTC.
fn expiry(text: &str) -> Result<Timestamp, &'static str> {
    text.parse::<Timestamp>()
        .or_else(|_| {
            text.parse::<DateTime>()
                .and_then(|at| at.to_zoned(TimeZone::UTC))
                .map(|at| at.timestamp())
        })
        .map_err(|_| "expires")
}

/// `key` as the core takes it. `Err` names a field that holds something no host step is asked
/// about. A blocked key is the core's to deny, whatever else it holds.
pub(super) fn read(key: &KeyEntry) -> Result<Keyed<Key>, &'static str> {
    let blocked = key.blocked == Some(true);
    let Held {
        unasked,
        budget,
        aliased,
    } = held(key);
    let metadata = Metadata::read(&key.metadata);
    let unread = metadata.is_none().then_some("metadata");
    if let Some(field) = unasked.or(unread).filter(|_| !blocked) {
        return Err(field);
    }
    let metadata = metadata.unwrap_or_default();
    let expires = key.expires.as_deref().map(expiry).transpose()?;
    let record = KeyRecord {
        token: Some(KeyHash::new(key.token.clone().ok_or("token")?)),
        blocked,
        expires,
        is_proxy_admin: false,
        user_id: key.user_id.clone(),
        team_id: key.team_id.clone(),
        team_object_permission_id: key.team_object_permission_id.clone(),
        project_id: key.project_id.clone(),
        last_refreshed_at: None,
    };
    Ok(Keyed {
        record: record.clone(),
        row: Key {
            record,
            models: Arc::from(key.models.as_slice()),
            limits: KeyLimits {
                rpm: key.rpm_limit,
                tpm: key.tpm_limit,
                max_parallel_requests: key.max_parallel_requests,
                model_rpm: ModelLimits::new(metadata.model_rpm_limit),
                model_tpm: ModelLimits::new(metadata.model_tpm_limit),
            },
            priority: metadata.priority.map(Arc::from),
            alias: key.key_alias.as_deref().map(Arc::from),
            budget,
            aliased,
        },
    })
}

#[cfg(test)]
mod tests {
    use litellm_python_compat::json;
    use litellm_redis_store::FromValue;
    use rstest::rstest;
    use serde_json::{Value as Fields, json};

    use super::*;

    /// `UserAPIKeyAuth.model_dump(mode="json")` of a key `/key/generate` made with no parameters.
    const CACHED: &str = r#"{
            "access_group_ids": [], "agent_id": null, "aliases": {}, "allowed_cache_controls": [],
            "allowed_model_region": null, "allowed_routes": [], "api_key": null, "auto_rotate": false,
            "blocked": null, "budget_duration": null, "budget_fallbacks": {}, "budget_id": null,
            "budget_limits": null, "budget_reset_at": null, "config": {},
            "created_at": "2026-11-19T10:00:00.123000Z", "created_by": "default_user_id",
            "created_by_user": null, "customer_hierarchy": null, "end_user_id": null,
            "end_user_max_budget": null, "end_user_model_max_budget": null,
            "end_user_model_rpm_limit": null, "end_user_model_tpm_limit": null,
            "end_user_object_permission": null, "end_user_rpm_limit": null, "end_user_tpd_limit": null,
            "end_user_tpm_limit": null, "expires": null, "is_session_token": false, "jwt_claims": null,
            "key_alias": null, "key_id": null, "key_name": "sk-...-key", "key_rotation_at": null,
            "key_type": "default", "last_active": null, "last_refreshed_at": 1763546400.5,
            "last_rotation_at": null, "litellm_budget_table": null, "max_budget": null,
            "max_parallel_requests": null, "metadata": {}, "model_max_budget": {}, "model_spend": {},
            "models": [], "object_permission": null, "object_permission_id": null, "org_id": null,
            "organization_alias": null, "organization_max_budget": null, "organization_metadata": null,
            "organization_rpm_limit": null, "organization_tpm_limit": null, "parent_otel_span": null,
            "permissions": {}, "project_alias": null, "project_id": null, "project_metadata": null,
            "request_route": null, "rotation_count": 0, "rotation_interval": null,
            "router_settings": {}, "rpm_limit": null, "rpm_limit_per_model": null,
            "settings_updated_at": null, "soft_budget": null, "soft_budget_cooldown": false,
            "spend": 0.0, "team_alias": null, "team_blocked": false, "team_id": null,
            "team_max_budget": null, "team_member": null, "team_member_rpm_limit": null,
            "team_member_spend": null, "team_member_tpm_limit": null, "team_metadata": null,
            "team_model_aliases": null, "team_model_max_budget": null, "team_models": [],
            "team_object_permission": null, "team_object_permission_id": null,
            "team_rpm_limit": null, "team_soft_budget": null, "team_spend": null,
            "team_tpd_limit": null, "team_tpm_limit": null,
            "token": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", "total_spend": 0.0,
            "tpd_limit": null, "tpm_limit": null, "tpm_limit_per_model": null,
            "updated_at": "2026-11-19T10:00:00.123000Z", "updated_by": "default_user_id", "user": null,
            "user_email": null, "user_id": null, "user_max_budget": null,
            "user_model_max_budget": null, "user_role": null, "user_rpm_limit": null,
            "user_spend": null, "user_tpm_limit": null
    }"#;

    /// The cached key with `change` set on it.
    fn cached(change: Fields) -> KeyEntry {
        let mut entry: Fields = serde_json::from_str(CACHED).unwrap();
        entry
            .as_object_mut()
            .unwrap()
            .extend(change.as_object().unwrap().clone());
        KeyEntry::from_value(&json::loads(&entry.to_string()).unwrap()).unwrap()
    }

    #[rstest]
    fn a_key_python_cached_as_generated_is_read() {
        let key = read(&cached(json!({}))).unwrap();

        assert_eq!(
            (
                key.row.record == key.record,
                key.record.token,
                key.record.blocked,
                key.record.expires,
                key.row.models.clone(),
                key.row.limits.clone(),
                key.row.priority.clone(),
                key.row.budget,
                key.row.aliased,
            ),
            (
                true,
                Some(KeyHash::new("a".repeat(64))),
                false,
                None,
                Arc::from([]),
                KeyLimits::default(),
                None,
                None,
                false,
            )
        );
    }

    #[rstest]
    #[case::api_key(json!({"api_key": "hashed"}))]
    #[case::key_id(json!({"key_id": "id"}))]
    #[case::last_refreshed_at(json!({"last_refreshed_at": 1.5}))]
    #[case::request_route(json!({"request_route": "/v1/messages"}))]
    #[case::user(json!({"user": {"user_id": "u"}}))]
    #[case::created_by_user(json!({"created_by_user": {"user_id": "u"}}))]
    #[case::user_email(json!({"user_email": "a@b.c"}))]
    #[case::user_spend(json!({"user_spend": 1.0}))]
    #[case::a_field_this_build_does_not_name_left_empty(json!({"future_limit": null}))]
    fn a_field_that_only_describes_the_key_or_holds_nothing_is_no_reason_to_decline(
        #[case] change: Fields,
    ) {
        assert!(read(&cached(change)).is_ok());
    }

    #[rstest]
    #[case::allowed_model_region(json!({"allowed_model_region": "eu"}), "allowed_model_region")]
    #[case::customer_hierarchy(json!({"customer_hierarchy": {"a": 1}}), "customer_hierarchy")]
    #[case::end_user_id(json!({"end_user_id": "e"}), "end_user_id")]
    #[case::end_user_max_budget(json!({"end_user_max_budget": 1.0}), "end_user_max_budget")]
    #[case::end_user_model_max_budget(json!({"end_user_model_max_budget": {"m": 1}}), "end_user_model_max_budget")]
    #[case::end_user_model_rpm_limit(json!({"end_user_model_rpm_limit": {"m": 1}}), "end_user_model_rpm_limit")]
    #[case::end_user_model_tpm_limit(json!({"end_user_model_tpm_limit": {"m": 1}}), "end_user_model_tpm_limit")]
    #[case::end_user_object_permission(json!({"end_user_object_permission": {"object_permission_id": "p"}}), "end_user_object_permission")]
    #[case::end_user_rpm_limit(json!({"end_user_rpm_limit": 1}), "end_user_rpm_limit")]
    #[case::end_user_tpd_limit(json!({"end_user_tpd_limit": 1}), "end_user_tpd_limit")]
    #[case::end_user_tpm_limit(json!({"end_user_tpm_limit": 1}), "end_user_tpm_limit")]
    #[case::is_session_token(json!({"is_session_token": true}), "is_session_token")]
    #[case::jwt_claims(json!({"jwt_claims": {"sub": "s"}}), "jwt_claims")]
    #[case::litellm_budget_table(json!({"litellm_budget_table": {"max_budget": 1}}), "litellm_budget_table")]
    #[case::object_permission(json!({"object_permission": {"object_permission_id": "p"}}), "object_permission")]
    #[case::project_metadata(json!({"project_metadata": {"a": 1}}), "project_metadata")]
    #[case::rpm_limit_per_model(json!({"rpm_limit_per_model": {"m": 1}}), "rpm_limit_per_model")]
    #[case::team_object_permission(json!({"team_object_permission": {"object_permission_id": "p"}}), "team_object_permission")]
    #[case::tpm_limit_per_model(json!({"tpm_limit_per_model": {"m": 1}}), "tpm_limit_per_model")]
    #[case::user_max_budget(json!({"user_max_budget": 1.0}), "user_max_budget")]
    #[case::user_model_max_budget(json!({"user_model_max_budget": {"m": 1}}), "user_model_max_budget")]
    #[case::user_role(json!({"user_role": "internal_user"}), "user_role")]
    #[case::user_rpm_limit(json!({"user_rpm_limit": 1}), "user_rpm_limit")]
    #[case::user_tpm_limit(json!({"user_tpm_limit": 1}), "user_tpm_limit")]
    #[case::a_field_this_build_does_not_name(json!({"future_limit": 1}), "unread")]
    fn an_auth_field_the_native_app_does_not_honor_is_named(
        #[case] change: Fields,
        #[case] field: &str,
    ) {
        assert_eq!(read(&cached(change)).err(), Some(field));
    }

    #[rstest]
    #[case::with_an_offset("2100-01-01T09:00:00+09:00")]
    #[case::as_utc("2100-01-01T00:00:00Z")]
    #[case::with_no_offset("2100-01-01T00:00:00")]
    fn an_expiry_is_read_as_python_wrote_it(#[case] expires: &str) {
        let key = read(&cached(json!({"expires": expires}))).unwrap();

        assert_eq!(
            key.record.expires,
            Some("2100-01-01T00:00:00Z".parse().unwrap())
        );
    }

    #[rstest]
    fn an_expiry_that_is_no_time_declines() {
        assert_eq!(
            read(&cached(json!({"expires": "tomorrow"}))).err(),
            Some("expires")
        );
    }
}
