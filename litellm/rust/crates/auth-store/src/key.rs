//! A virtual key as the database holds it, in the shape the auth cache holds it.

use std::sync::Arc;

use chrono::SecondsFormat;
use litellm_gateway_auth_core::KeyHash;
use litellm_redis_store::auth_cache::{KeyEntry, ObjectPermissionEntry};
use litellm_repositories::{KeyRepository, KeyView, ObjectPermissionRepository, Side};

use crate::{
    Error,
    column::{column, nullable, time},
    member::team_member_column,
    object_permission::ObjectPermissions,
};

/// The key as the database holds it, in the shape the auth cache holds it: the `combined_view`
/// row with the object permission the key points at. These are the database steps of a key
/// read, apart from the cache steps around them, so a host that bounds its database reads with
/// a policy of its own wraps exactly these.
pub struct KeyDatabase {
    keys: Arc<KeyRepository>,
    object_permissions: ObjectPermissions,
}

impl KeyDatabase {
    pub fn new(
        keys: Arc<KeyRepository>,
        object_permissions: Arc<ObjectPermissionRepository>,
    ) -> Self {
        Self {
            keys,
            object_permissions: ObjectPermissions::new(object_permissions),
        }
    }

    /// The key whose hash is `hash` with nothing nested, or `None` when no key has it. A hash
    /// rotated away from a key within its grace period answers with the active key. One query,
    /// so a host bounds it alone and reads the object permission under a bound of its own.
    pub async fn key(&self, hash: &KeyHash, side: Side) -> Result<Option<KeyEntry>, Error> {
        let Some(found) = self.keys.find(hash.as_str(), side).await? else {
            return Ok(None);
        };
        key_entry(found.view(), None, found.last_refreshed_at).map(Some)
    }

    /// [`Self::key`] with the object permission it points at, for a host with no bound of its
    /// own to put between the two reads. A permission row that is gone leaves `object_permission`
    /// empty on either side, for the caller to refuse.
    pub async fn read(&self, hash: &KeyHash, side: Side) -> Result<Option<KeyEntry>, Error> {
        let Some(key) = self.key(hash, side).await? else {
            return Ok(None);
        };
        let object_permission = self
            .object_permissions
            .nested(key.object_permission_id.as_deref(), side)
            .await?;
        Ok(Some(KeyEntry {
            object_permission,
            ..key
        }))
    }
}

/// The entry `main`'s `get_key_object` cached for the row, which Python's `cache_key_object` still
/// writes for every key a management endpoint changes: the row through the view model with its
/// nulls dropped, then the full `UserAPIKeyAuth` dump. The paired cases in this crate's
/// `tests/key.rs` and `tests/unit/redis/stores/test_auth_cache_key.py` pin the two to the same
/// bytes.
///
/// A column that is null takes the model's default, which is not null for `allowed_cache_controls`,
/// `allowed_routes`, `rotation_count` and `auto_rotate`. `expires` is the text the repository hands
/// Python, since the model keeps a string there. The team member is what `Member` dumps. Fields
/// only a cache entry carries are unset.
fn key_entry(
    view: &KeyView,
    object_permission: Option<ObjectPermissionEntry>,
    last_refreshed_at: f64,
) -> Result<KeyEntry, Error> {
    let KeyView {
        token,
        key_name,
        key_alias,
        soft_budget_cooldown,
        spend,
        total_spend,
        expires,
        models,
        aliases,
        config,
        router_settings,
        user_id,
        team_id,
        agent_id,
        project_id,
        permissions,
        max_parallel_requests,
        metadata,
        blocked,
        tpm_limit,
        rpm_limit,
        tpd_limit,
        max_budget,
        soft_budget,
        budget_duration,
        budget_reset_at,
        allowed_cache_controls,
        allowed_routes,
        key_type,
        access_group_ids,
        model_spend,
        model_max_budget,
        budget_fallbacks,
        budget_id,
        org_id,
        object_permission_id,
        created_at,
        created_by,
        updated_at,
        updated_by,
        settings_updated_at,
        last_active,
        rotation_count,
        auto_rotate,
        rotation_interval,
        last_rotation_at,
        key_rotation_at,
        budget_limits,
        team_spend,
        team_alias,
        team_tpm_limit,
        team_rpm_limit,
        team_tpd_limit,
        team_max_budget,
        team_soft_budget,
        team_model_max_budget,
        team_models,
        team_blocked,
        team_model_aliases,
        team_member,
        team_metadata,
        team_object_permission_id,
        team_member_spend,
        team_member_tpm_limit,
        team_member_rpm_limit,
        organization_alias,
        organization_max_budget,
        organization_tpm_limit,
        organization_rpm_limit,
        organization_metadata,
        project_alias,
    } = view;
    Ok(KeyEntry {
        key_id: None,
        token: Some(token.clone()),
        key_name: key_name.clone(),
        key_alias: key_alias.clone(),
        spend: *spend,
        total_spend: *total_spend,
        max_budget: *max_budget,
        expires: expires.map(|at| at.and_utc().to_rfc3339_opts(SecondsFormat::AutoSi, false)),
        models: models.clone(),
        aliases: column("aliases", aliases)?,
        config: column("config", config)?,
        user_id: user_id.clone(),
        team_id: team_id.clone(),
        agent_id: agent_id.clone(),
        project_id: project_id.clone(),
        max_parallel_requests: max_parallel_requests.map(i64::from),
        metadata: column("metadata", metadata)?,
        tpm_limit: *tpm_limit,
        rpm_limit: *rpm_limit,
        tpd_limit: *tpd_limit,
        budget_duration: budget_duration.clone(),
        budget_reset_at: time("budget_reset_at", *budget_reset_at)?,
        allowed_cache_controls: Some(allowed_cache_controls.clone().unwrap_or_default()),
        allowed_routes: Some(allowed_routes.clone().unwrap_or_default()),
        key_type: key_type.clone(),
        permissions: column("permissions", permissions)?,
        model_spend: column("model_spend", model_spend)?,
        model_max_budget: column("model_max_budget", model_max_budget)?,
        budget_fallbacks: column("budget_fallbacks", budget_fallbacks)?,
        soft_budget_cooldown: *soft_budget_cooldown,
        blocked: *blocked,
        litellm_budget_table: None,
        budget_id: budget_id.clone(),
        org_id: org_id.clone(),
        created_at: time("created_at", *created_at)?,
        created_by: created_by.clone(),
        updated_at: time("updated_at", *updated_at)?,
        updated_by: updated_by.clone(),
        settings_updated_at: time("settings_updated_at", *settings_updated_at)?,
        last_active: time("last_active", *last_active)?,
        object_permission_id: object_permission_id.clone(),
        object_permission,
        access_group_ids: access_group_ids.clone(),
        rotation_count: Some(rotation_count.map_or(0, i64::from)),
        auto_rotate: Some(auto_rotate.unwrap_or(false)),
        rotation_interval: rotation_interval.clone(),
        last_rotation_at: time("last_rotation_at", *last_rotation_at)?,
        key_rotation_at: time("key_rotation_at", *key_rotation_at)?,
        router_settings: nullable("router_settings", router_settings.as_ref())?,
        budget_limits: nullable("budget_limits", budget_limits.as_ref())?,
        team_spend: *team_spend,
        team_alias: team_alias.clone(),
        team_tpm_limit: *team_tpm_limit,
        team_rpm_limit: *team_rpm_limit,
        team_tpd_limit: *team_tpd_limit,
        team_max_budget: *team_max_budget,
        team_soft_budget: *team_soft_budget,
        team_model_max_budget: nullable("team_model_max_budget", team_model_max_budget.as_ref())?,
        team_models: team_models.clone(),
        team_blocked: *team_blocked,
        soft_budget: *soft_budget,
        team_model_aliases: nullable("team_model_aliases", team_model_aliases.as_ref())?,
        team_member: team_member.as_ref().map(team_member_column).transpose()?,
        team_metadata: nullable("team_metadata", team_metadata.as_ref())?,
        team_object_permission_id: team_object_permission_id.clone(),
        team_member_spend: *team_member_spend,
        team_member_tpm_limit: *team_member_tpm_limit,
        team_member_rpm_limit: *team_member_rpm_limit,
        end_user_id: None,
        end_user_tpm_limit: None,
        end_user_rpm_limit: None,
        end_user_tpd_limit: None,
        end_user_max_budget: None,
        end_user_model_max_budget: None,
        end_user_model_tpm_limit: None,
        end_user_model_rpm_limit: None,
        customer_hierarchy: None,
        organization_alias: organization_alias.clone(),
        organization_max_budget: *organization_max_budget,
        organization_tpm_limit: *organization_tpm_limit,
        organization_rpm_limit: *organization_rpm_limit,
        organization_metadata: nullable("organization_metadata", organization_metadata.as_ref())?,
        project_alias: project_alias.clone(),
        project_metadata: None,
        last_refreshed_at: Some(last_refreshed_at),
        api_key: None,
        user_role: None,
        allowed_model_region: None,
        parent_otel_span: None,
        rpm_limit_per_model: None,
        tpm_limit_per_model: None,
        user_tpm_limit: None,
        user_rpm_limit: None,
        user_email: None,
        user_spend: None,
        user_max_budget: None,
        user_model_max_budget: None,
        request_route: None,
        is_session_token: false,
        user: None,
        created_by_user: None,
        end_user_object_permission: None,
        team_object_permission: None,
        jwt_claims: None,
        unread: Vec::new(),
    })
}
