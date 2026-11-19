//! A key as Python's `UserApiKeyCache` holds it with `enable_redis_auth_cache` on: the
//! `UserAPIKeyAuth.model_dump(mode="json")` its auth flow wrote under the key's hash. The auth
//! fields beyond the key's row are taken apart by name here, and the rest must be exactly the
//! row, so a field Python adds is a miss that reads the database, never a key served unchecked.

use litellm_python_compat::json;
use litellm_redis_store::{Send, auth_cache::AuthCache};
use litellm_repositories::{KeyView, RawJson};
use serde::Deserialize;
use serde_json::Map;

use super::{Carried, KeyHash};

/// Auth fields that only describe the key, or the request that cached it.
const DESCRIBING: [&str; 9] = [
    "api_key",
    "created_by_user",
    "key_id",
    "last_refreshed_at",
    "parent_otel_span",
    "request_route",
    "user",
    "user_email",
    "user_spend",
];

#[derive(Deserialize)]
struct CachedKey {
    #[serde(flatten)]
    row: Map<String, serde_json::Value>,
    allowed_model_region: Option<String>,
    customer_hierarchy: Option<RawJson>,
    end_user_id: Option<String>,
    end_user_max_budget: Option<f64>,
    end_user_model_max_budget: Option<RawJson>,
    end_user_model_rpm_limit: Option<RawJson>,
    end_user_model_tpm_limit: Option<RawJson>,
    end_user_object_permission: Option<RawJson>,
    end_user_rpm_limit: Option<i64>,
    end_user_tpd_limit: Option<i64>,
    end_user_tpm_limit: Option<i64>,
    #[serde(default)]
    is_session_token: bool,
    jwt_claims: Option<RawJson>,
    litellm_budget_table: Option<RawJson>,
    object_permission: Option<RawJson>,
    project_metadata: Option<RawJson>,
    rpm_limit_per_model: Option<RawJson>,
    team_object_permission: Option<RawJson>,
    tpm_limit_per_model: Option<RawJson>,
    user_max_budget: Option<f64>,
    user_model_max_budget: Option<RawJson>,
    user_role: Option<String>,
    user_rpm_limit: Option<i64>,
    user_tpm_limit: Option<i64>,
}

/// The key Python cached and the first auth field it holds that the native app does not honor.
pub(super) struct Cached {
    pub row: KeyView,
    pub carried: Option<&'static str>,
}

impl Cached {
    fn decode(text: &str) -> Result<Self, serde_json::Error> {
        let key: CachedKey = serde_json::from_str(text)?;
        let carried = key.carried();
        let row: Map<_, _> = key
            .row
            .into_iter()
            .filter(|(field, _)| !DESCRIBING.contains(&field.as_str()))
            .collect();
        let row = serde_json::from_str(&serde_json::to_string(&row)?)?;
        Ok(Self { row, carried })
    }
}

impl CachedKey {
    fn carried(&self) -> Option<&'static str> {
        let Self {
            row: _,
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
        } = self;
        carried!(
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
        )
    }
}

/// The key Python cached under `hash`, read from Redis alone, since this process hears no
/// evictions and a copy in its memory could outlive a block. `None` when Redis does not hold it,
/// fails, or holds what this gateway cannot read, so the database answers instead.
pub(super) async fn read(cache: &AuthCache, hash: &KeyHash) -> Option<Cached> {
    let values = cache
        .get_many(Send::Now, &[hash.as_str().to_owned()])?
        .await
        .inspect_err(|error| tracing::warn!(%error, "the auth cache was not read from Redis"))
        .ok()?;
    let value = values.into_iter().next().flatten()?;
    let text = json::dumps(&value)
        .inspect_err(|error| tracing::debug!(%error, "a cached key is not JSON"))
        .ok()?;
    Cached::decode(&text)
        .inspect_err(|error| tracing::debug!(%error, "a cached key is not one this gateway reads"))
        .ok()
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use serde_json::{Value, json};

    use super::*;

    /// `UserAPIKeyAuth.model_dump(mode="json")` of a key `/key/generate` made with no parameters.
    fn python_cached_key() -> Value {
        serde_json::from_str(
            r#"{
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
    }"#,
        )
        .unwrap()
    }

    fn decoded(change: Value) -> Result<Cached, serde_json::Error> {
        let mut cached = python_cached_key();
        cached
            .as_object_mut()
            .unwrap()
            .extend(change.as_object().unwrap().clone());
        Cached::decode(&cached.to_string())
    }

    #[test]
    fn a_key_python_cached_as_generated_carries_nothing() {
        let cached = decoded(json!({})).unwrap();

        assert_eq!(cached.carried, None);
        assert_eq!(cached.row.token, "a".repeat(64));
        assert_eq!(
            cached.row.created_at,
            Some("2026-11-19T10:00:00.123".parse().unwrap())
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
    fn an_auth_field_that_only_describes_the_key_carries_nothing(#[case] change: Value) {
        assert_eq!(decoded(change).unwrap().carried, None);
    }

    #[rstest]
    #[case::allowed_model_region(json!({"allowed_model_region": "eu"}), "allowed_model_region")]
    #[case::customer_hierarchy(json!({"customer_hierarchy": {"a": 1}}), "customer_hierarchy")]
    #[case::end_user_id(json!({"end_user_id": "e"}), "end_user_id")]
    #[case::end_user_max_budget(json!({"end_user_max_budget": 1.0}), "end_user_max_budget")]
    #[case::end_user_model_max_budget(json!({"end_user_model_max_budget": {"m": 1}}), "end_user_model_max_budget")]
    #[case::end_user_model_rpm_limit(json!({"end_user_model_rpm_limit": {"m": 1}}), "end_user_model_rpm_limit")]
    #[case::end_user_model_tpm_limit(json!({"end_user_model_tpm_limit": {"m": 1}}), "end_user_model_tpm_limit")]
    #[case::end_user_object_permission(json!({"end_user_object_permission": {"a": 1}}), "end_user_object_permission")]
    #[case::end_user_rpm_limit(json!({"end_user_rpm_limit": 1}), "end_user_rpm_limit")]
    #[case::end_user_tpd_limit(json!({"end_user_tpd_limit": 1}), "end_user_tpd_limit")]
    #[case::end_user_tpm_limit(json!({"end_user_tpm_limit": 1}), "end_user_tpm_limit")]
    #[case::is_session_token(json!({"is_session_token": true}), "is_session_token")]
    #[case::jwt_claims(json!({"jwt_claims": {"sub": "s"}}), "jwt_claims")]
    #[case::litellm_budget_table(json!({"litellm_budget_table": {"max_budget": 1}}), "litellm_budget_table")]
    #[case::object_permission(json!({"object_permission": {"a": 1}}), "object_permission")]
    #[case::project_metadata(json!({"project_metadata": {"a": 1}}), "project_metadata")]
    #[case::rpm_limit_per_model(json!({"rpm_limit_per_model": {"m": 1}}), "rpm_limit_per_model")]
    #[case::team_object_permission(json!({"team_object_permission": {"a": 1}}), "team_object_permission")]
    #[case::tpm_limit_per_model(json!({"tpm_limit_per_model": {"m": 1}}), "tpm_limit_per_model")]
    #[case::user_max_budget(json!({"user_max_budget": 1.0}), "user_max_budget")]
    #[case::user_model_max_budget(json!({"user_model_max_budget": {"m": 1}}), "user_model_max_budget")]
    #[case::user_role(json!({"user_role": "internal_user"}), "user_role")]
    #[case::user_rpm_limit(json!({"user_rpm_limit": 1}), "user_rpm_limit")]
    #[case::user_tpm_limit(json!({"user_tpm_limit": 1}), "user_tpm_limit")]
    fn an_auth_field_the_native_app_does_not_honor_is_named(
        #[case] change: Value,
        #[case] field: &str,
    ) {
        assert_eq!(decoded(change).unwrap().carried, Some(field));
    }

    #[rstest]
    #[case::an_unknown_field(json!({"future_limit": 1}))]
    #[case::a_column_of_another_type(json!({"spend": "free"}))]
    #[case::a_time_it_cannot_read(json!({"expires": "tomorrow"}))]
    fn a_key_this_gateway_cannot_read_is_refused(#[case] change: Value) {
        assert!(decoded(change).is_err());
    }
}
