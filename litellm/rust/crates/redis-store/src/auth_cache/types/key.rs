use jiff::Timestamp;
use litellm_python_compat::Value;

use super::{
    bag::{Bag, Lists},
    member::CachedMember,
    object_permission::CachedObjectPermission,
    record::record,
};

record! {
    /// `UserAPIKeyAuth.model_dump(mode="json")`, the value under
    /// [`AuthObject::Key`](crate::auth_cache::AuthObject::Key). A key read from the database is
    /// cached in this shape too, so it is what a key read returns from either tier. `expires`
    /// stays the text it was written as, since the model types it `str | datetime` and Python
    /// before 3.11 cannot read a `Z` suffix back.
    pub struct CachedKey {
        key_id: Option<String>,
        token: Option<String>,
        key_name: Option<String>,
        key_alias: Option<String>,
        spend: f64,
        total_spend: f64,
        max_budget: Option<f64>,
        expires: Option<String>,
        models: Vec<String>,
        aliases: Bag,
        config: Bag,
        user_id: Option<String>,
        team_id: Option<String>,
        agent_id: Option<String>,
        project_id: Option<String>,
        max_parallel_requests: Option<i64>,
        metadata: Bag,
        tpm_limit: Option<i64>,
        rpm_limit: Option<i64>,
        tpd_limit: Option<i64>,
        budget_duration: Option<String>,
        budget_reset_at: Option<Timestamp>,
        allowed_cache_controls: Option<Vec<String>> = Some(Vec::new()),
        allowed_routes: Option<Vec<String>> = Some(Vec::new()),
        key_type: Option<String>,
        permissions: Bag,
        model_spend: Bag,
        model_max_budget: Bag,
        budget_fallbacks: Lists,
        soft_budget_cooldown: bool,
        blocked: Option<bool>,
        litellm_budget_table: Option<Bag>,
        budget_id: Option<String>,
        org_id: Option<String>,
        created_at: Option<Timestamp>,
        created_by: Option<String>,
        updated_at: Option<Timestamp>,
        updated_by: Option<String>,
        settings_updated_at: Option<Timestamp>,
        last_active: Option<Timestamp>,
        object_permission_id: Option<String>,
        object_permission: Option<CachedObjectPermission>,
        access_group_ids: Option<Vec<String>>,
        rotation_count: Option<i64> = Some(0),
        auto_rotate: Option<bool> = Some(false),
        rotation_interval: Option<String>,
        last_rotation_at: Option<Timestamp>,
        key_rotation_at: Option<Timestamp>,
        router_settings: Option<Bag>,
        budget_limits: Option<Vec<Bag>>,
        team_spend: Option<f64>,
        team_alias: Option<String>,
        team_tpm_limit: Option<i64>,
        team_rpm_limit: Option<i64>,
        team_tpd_limit: Option<i64>,
        team_max_budget: Option<f64>,
        team_soft_budget: Option<f64>,
        team_model_max_budget: Option<Bag>,
        team_models: Vec<String>,
        team_blocked: bool,
        soft_budget: Option<f64>,
        team_model_aliases: Option<Bag>,
        team_member: Option<CachedMember>,
        team_metadata: Option<Bag>,
        team_object_permission_id: Option<String>,
        team_member_spend: Option<f64>,
        team_member_tpm_limit: Option<i64>,
        team_member_rpm_limit: Option<i64>,
        end_user_id: Option<String>,
        end_user_tpm_limit: Option<i64>,
        end_user_rpm_limit: Option<i64>,
        end_user_tpd_limit: Option<i64>,
        end_user_max_budget: Option<f64>,
        end_user_model_max_budget: Option<Bag>,
        end_user_model_tpm_limit: Option<Bag>,
        end_user_model_rpm_limit: Option<Bag>,
        #[redact(with = "[REDACTED]")]
        customer_hierarchy: Option<Value>,
        organization_alias: Option<String>,
        organization_max_budget: Option<f64>,
        organization_tpm_limit: Option<i64>,
        organization_rpm_limit: Option<i64>,
        organization_metadata: Option<Bag>,
        project_alias: Option<String>,
        project_metadata: Option<Bag>,
        last_refreshed_at: Option<f64>,
        api_key: Option<String>,
        user_role: Option<String>,
        allowed_model_region: Option<String>,
        #[redact(with = "[REDACTED]")]
        parent_otel_span: Option<Value>,
        rpm_limit_per_model: Option<Bag>,
        tpm_limit_per_model: Option<Bag>,
        user_tpm_limit: Option<i64>,
        user_rpm_limit: Option<i64>,
        user_email: Option<String>,
        user_spend: Option<f64>,
        user_max_budget: Option<f64>,
        user_model_max_budget: Option<Bag>,
        request_route: Option<String>,
        is_session_token: bool,
        #[redact(with = "[REDACTED]")]
        user: Option<Value>,
        #[redact(with = "[REDACTED]")]
        created_by_user: Option<Value>,
        end_user_object_permission: Option<CachedObjectPermission>,
        team_object_permission: Option<CachedObjectPermission>,
        jwt_claims: Option<Bag>,
    }
}

#[cfg(test)]
mod tests {
    use jiff::Timestamp;
    use litellm_python_compat::{Value, json};
    use rstest::rstest;

    use super::CachedKey;
    use crate::{DecodeError, FromValue, IntoValue};

    fn key(entry: &str) -> Result<CachedKey, DecodeError> {
        CachedKey::from_value(&json::loads(entry).unwrap())
    }

    #[rstest]
    fn a_field_the_entry_lacks_takes_the_model_default() {
        let key = key(r#"{"token": "hashed"}"#).unwrap();

        assert_eq!(
            (
                key.allowed_cache_controls,
                key.allowed_routes,
                key.rotation_count,
                key.auto_rotate,
                key.aliases.is_empty(),
                key.access_group_ids,
                key.router_settings,
                key.spend,
                key.team_blocked,
            ),
            (
                Some(Vec::new()),
                Some(Vec::new()),
                Some(0),
                Some(false),
                true,
                None,
                None,
                0.0,
                false,
            )
        );
    }

    #[rstest]
    fn a_field_this_build_does_not_name_is_kept_and_written_after_the_named_ones() {
        let key =
            key(r#"{"future_limit": {"rpm": 7}, "token": "hashed", "also_new": null}"#).unwrap();

        let unread = key.unread.clone();
        let written = json::dumps(&key.into_value()).unwrap();

        assert_eq!(
            (
                unread,
                written.starts_with(r#"{"key_id": null, "token": "hashed", "#),
                written.ends_with(
                    r#""jwt_claims": null, "future_limit": {"rpm": 7}, "also_new": null}"#
                ),
            ),
            (
                vec![
                    (
                        "future_limit".to_owned(),
                        json::loads(r#"{"rpm": 7}"#).unwrap()
                    ),
                    ("also_new".to_owned(), Value::None),
                ],
                true,
                true,
            )
        );
    }

    #[rstest]
    #[case::utc_as_pydantic_writes_it(
        r#""2026-11-19T10:00:00.123000Z""#,
        "2026-11-19T10:00:00.123Z"
    )]
    #[case::whole_seconds(r#""2026-11-19T10:00:00Z""#, "2026-11-19T10:00:00Z")]
    #[case::another_offset(r#""2026-11-19T19:00:00+09:00""#, "2026-11-19T10:00:00Z")]
    #[case::no_offset_is_utc(r#""2026-11-19T10:00:00.5""#, "2026-11-19T10:00:00.5Z")]
    fn a_time_python_wrote_is_its_instant(#[case] written: &str, #[case] instant: &str) {
        let key = key(&format!(r#"{{"created_at": {written}}}"#)).unwrap();

        assert_eq!(key.created_at, Some(instant.parse::<Timestamp>().unwrap()));
    }

    #[rstest]
    #[case::a_fraction("2026-11-19T10:00:00.123Z", r#""2026-11-19T10:00:00.123000Z""#)]
    #[case::whole_seconds("2026-11-19T10:00:00Z", r#""2026-11-19T10:00:00Z""#)]
    #[case::before_the_epoch("1969-12-31T23:59:59.5Z", r#""1969-12-31T23:59:59.500000Z""#)]
    fn a_time_is_written_as_pydantic_writes_an_aware_utc_datetime(
        #[case] instant: &str,
        #[case] written: &str,
    ) {
        let key = CachedKey {
            created_at: Some(instant.parse().unwrap()),
            ..key("{}").unwrap()
        };

        let entry = json::dumps(&key.into_value()).unwrap();

        assert!(
            entry.contains(&format!(r#""created_at": {written},"#)),
            "{entry}"
        );
    }

    #[rstest]
    #[case::a_number_that_is_text(r#"{"spend": "free"}"#, "spend")]
    #[case::a_required_number_that_is_null(r#"{"spend": null}"#, "spend")]
    #[case::a_limit_that_is_a_fraction(r#"{"rpm_limit": 1.5}"#, "rpm_limit")]
    #[case::a_time_that_is_no_time(r#"{"created_at": "tomorrow"}"#, "created_at")]
    #[case::a_model_list_holding_a_number(r#"{"models": ["a", 1]}"#, "models")]
    #[case::a_member_without_a_role(r#"{"team_member": {"user_id": "u"}}"#, "team_member")]
    #[case::a_member_with_another_role(
        r#"{"team_member": {"user_id": "u", "role": "owner"}}"#,
        "team_member"
    )]
    #[case::metadata_that_is_a_list(r#"{"metadata": []}"#, "metadata")]
    #[case::metadata_that_is_null(r#"{"metadata": null}"#, "metadata")]
    #[case::fallbacks_that_are_not_lists(r#"{"budget_fallbacks": {"m": "n"}}"#, "budget_fallbacks")]
    #[case::budget_limits_that_are_no_list(r#"{"budget_limits": {"m": 1}}"#, "budget_limits")]
    #[case::router_settings_that_are_text(r#"{"router_settings": "{}"}"#, "router_settings")]
    #[case::a_permission_whose_tool_map_is_text(
        r#"{"object_permission": {"object_permission_id": "p", "mcp_tool_permissions": "{\"s\": [\"t\"]}"}}"#,
        "object_permission"
    )]
    #[case::a_permission_without_its_id(
        r#"{"object_permission": {"models": []}}"#,
        "object_permission"
    )]
    fn an_entry_the_model_would_refuse_does_not_decode(#[case] entry: &str, #[case] field: &str) {
        assert!(
            matches!(key(entry), Err(DecodeError::Shape(refused)) if refused == field),
            "{:?}",
            key(entry).err()
        );
    }

    #[rstest]
    fn a_float_field_takes_the_integer_python_also_takes() {
        assert_eq!(
            key(r#"{"spend": 3, "max_budget": 10}"#)
                .map(|key| (key.spend, key.max_budget))
                .unwrap(),
            (3.0, Some(10.0))
        );
    }

    #[rstest]
    fn debug_shows_no_settings_bag() {
        let key = key(concat!(
            r#"{"token": "hashed", "metadata": {"api_key": "credential-value"}, "#,
            r#""team_member": {"user_id": "u", "role": "user", "note": "credential-value"}, "#,
            r#""router_settings": {"redis_password": "credential-value"}, "budget_limits": [{"k": "credential-value"}], "#,
            r#""later_field": "credential-value"}"#,
        ))
        .unwrap();

        let debug = format!("{key:?}");

        assert_eq!(
            (
                debug.contains("credential-value"),
                debug.contains("[REDACTED]"),
                debug.contains("hashed"),
            ),
            (false, true, true)
        );
    }
}
