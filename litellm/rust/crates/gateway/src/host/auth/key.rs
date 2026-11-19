//! A virtual key's row, as far as the native host reads it. Each column is taken apart by name,
//! so a column added to the row does not compile here until it is honored, left to a host step or
//! named as one no step is asked about.

use std::{collections::BTreeMap, sync::Arc, time::SystemTime};

use chrono::NaiveDateTime;
use jiff::Timestamp;
use litellm_gateway_auth::keys::{HeldKey, KeyLimits, ModelLimits};
use litellm_gateway_auth_core::{
    KeyHash,
    authn::key::{KeyRecord, Keyed},
};
use litellm_repositories::{KeyView, RawJson};
use serde::Deserialize;

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

/// Whether `Permissions::Models` reads a `models` entry as the proxy does: a name, an access
/// group or `*`. A pattern like `anthropic/*` the proxy matches against providers is not read.
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

carried_when_held!(String, i32, i64, f64, NaiveDateTime);

impl Carried for Vec<String> {
    fn is_carried(&self) -> bool {
        !self.is_empty()
    }
}

impl Carried for RawJson {
    fn is_carried(&self) -> bool {
        !matches!(self.get().trim(), "{}" | "[]" | "null")
    }
}

impl<T: Carried> Carried for Option<T> {
    fn is_carried(&self) -> bool {
        self.as_ref().is_some_and(Carried::is_carried)
    }
}

/// The team the core reads no team row for, so no step declines a key in it.
const UI_TEAM: &str = "litellm-dashboard";

/// A virtual key the native host read, with what its routes and the host's own steps read of it.
#[derive(Clone, Debug, PartialEq)]
pub struct Key {
    expires_at: Option<SystemTime>,
    models: Arc<[String]>,
    limits: KeyLimits,
    /// The first budget column the row holds, which `key_budgets` declines.
    pub(super) budget: Option<&'static str>,
    /// The row maps model names, which `normalize_model` declines.
    pub(super) aliased: bool,
}

impl HeldKey for Key {
    fn expires_at(&self) -> Option<SystemTime> {
        self.expires_at
    }

    fn models(&self) -> Arc<[String]> {
        self.models.clone()
    }

    fn limits(&self) -> KeyLimits {
        self.limits.clone()
    }
}

/// The `metadata` the native app honors: the key's own limit for each model it names one for,
/// which `/key/generate` stores from `model_rpm_limit` and `model_tpm_limit`.
#[derive(Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct Metadata {
    #[serde(default)]
    model_rpm_limit: BTreeMap<String, i64>,
    #[serde(default)]
    model_tpm_limit: BTreeMap<String, i64>,
}

impl Metadata {
    /// `None` when `metadata` holds anything else, or a limit that is not a whole number.
    fn read(metadata: &RawJson) -> Option<Self> {
        if !metadata.is_carried() {
            return Some(Self::default());
        }
        serde_json::from_str(metadata.get()).ok()
    }
}

/// What the row holds beside the columns the core's record carries.
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
/// other column that holds something is one no step is asked about. The team, organization and
/// project columns are only filled for a key that names one.
fn held(key: &KeyView) -> Held {
    let KeyView {
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
    } = key;
    let other_key_type = key_type
        .as_deref()
        .is_some_and(|key_type| key_type != DEFAULT_KEY_TYPE);
    let unread_models = models.iter().any(|model| !is_read(model));
    let ui_team = team_id.as_deref() == Some(UI_TEAM);
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

/// Whether the native app answers for `key` with no host step declining it: it is not blocked,
/// names no owner, team or project, and holds no budget or alias.
pub(super) fn is_answered_natively(key: &Keyed<Key>) -> bool {
    let KeyRecord {
        blocked,
        user_id,
        team_id,
        project_id,
        ..
    } = &key.record;
    !blocked
        && user_id.is_none()
        && team_id.is_none()
        && project_id.is_none()
        && key.row.budget.is_none()
        && !key.row.aliased
}

/// `key` as the core takes it. `Err` names a column that holds something no host step is asked
/// about, or a cached auth field in `also` that does. A blocked key is the core's to deny,
/// whatever else it holds.
pub(super) fn read(key: &KeyView, also: Option<&'static str>) -> Result<Keyed<Key>, &'static str> {
    let blocked = key.blocked == Some(true);
    let Held {
        unasked,
        budget,
        aliased,
    } = held(key);
    let metadata = Metadata::read(&key.metadata);
    let unread = metadata.is_none().then_some("metadata");
    if let Some(column) = unasked.or(unread).or(also).filter(|_| !blocked) {
        return Err(column);
    }
    let metadata = metadata.unwrap_or_default();
    let expires = key
        .expires
        .map(|at| Timestamp::from_microsecond(at.and_utc().timestamp_micros()))
        .transpose()
        .map_err(|_| "expires")?;
    Ok(Keyed {
        record: KeyRecord {
            token: Some(KeyHash::new(key.token.clone())),
            blocked,
            expires,
            is_proxy_admin: false,
            user_id: key.user_id.clone(),
            team_id: key.team_id.clone(),
            team_object_permission_id: key.team_object_permission_id.clone(),
            project_id: key.project_id.clone(),
            last_refreshed_at: None,
        },
        row: Key {
            expires_at: key.expires.map(|at| SystemTime::from(at.and_utc())),
            models: Arc::from(key.models.as_slice()),
            limits: KeyLimits {
                rpm: key.rpm_limit,
                tpm: key.tpm_limit,
                max_parallel_requests: key.max_parallel_requests.map(i64::from),
                model_rpm: ModelLimits::new(metadata.model_rpm_limit),
                model_tpm: ModelLimits::new(metadata.model_tpm_limit),
            },
            budget,
            aliased,
        },
    })
}
