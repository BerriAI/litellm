//! The virtual keys the native app authenticates, read from the proxy's database on every request.
//!
//! The native app honors a key's expiry and its block and nothing else a key can carry, so it
//! only authenticates a key that carries nothing else. Each column of the key's row is taken apart
//! by name, so a column added to the row does not compile here until it is honored or named.

use std::time::SystemTime;

use chrono::NaiveDateTime;
use litellm_error::Source;
use litellm_gateway_auth::{
    Error,
    keys::{KeyHash, KeyLookup, KeyStatus, LookupFuture},
};
use litellm_repositories::{KeyRepository, KeyView, RawJson, Side};

/// The key type the proxy gives a key generated without one, which reaches every route.
const DEFAULT_KEY_TYPE: &str = "default";

trait Carried {
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

/// The first column of `key` that holds something the native app does not honor. The team,
/// organization and project columns are only filled for a key that names one.
fn carried(key: &KeyView) -> Option<&'static str> {
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
        key_type,
        soft_budget_cooldown,
        models,
        aliases,
        config,
        router_settings,
        user_id,
        team_id,
        agent_id,
        project_id,
        org_id,
        permissions,
        max_parallel_requests,
        metadata,
        tpm_limit,
        rpm_limit,
        tpd_limit,
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
    macro_rules! named {
        ($($column:ident),+ $(,)?) => {
            [$((stringify!($column), Carried::is_carried($column)),)+]
        };
    }
    let other_key_type = key_type
        .as_deref()
        .is_some_and(|key_type| key_type != DEFAULT_KEY_TYPE);
    [("key_type", other_key_type)]
        .into_iter()
        .chain(named![
            soft_budget_cooldown,
            models,
            aliases,
            config,
            router_settings,
            user_id,
            team_id,
            agent_id,
            project_id,
            org_id,
            permissions,
            max_parallel_requests,
            metadata,
            tpm_limit,
            rpm_limit,
            tpd_limit,
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
        ])
        .find(|(_, carried)| *carried)
        .map(|(column, _)| column)
}

/// `Err` names the column that holds something the native app does not honor.
fn status(key: &KeyView) -> Result<KeyStatus, &'static str> {
    if key.blocked == Some(true) {
        return Ok(KeyStatus::Revoked);
    }
    match carried(key) {
        Some(column) => Err(column),
        None => Ok(KeyStatus::Active {
            expires_at: key.expires.map(|at| SystemTime::from(at.and_utc())),
        }),
    }
}

/// The proxy's keys, each read from its database when asked for.
pub(crate) struct StoredKeys(pub KeyRepository);

impl KeyLookup for StoredKeys {
    fn lookup<'a>(&'a self, hash: &'a KeyHash) -> LookupFuture<'a> {
        Box::pin(async move {
            let found = self
                .0
                .find(hash.as_str(), Side::Read)
                .await
                .map_err(|error| Error::Unavailable(Source::new(error)))?;
            found
                .map(|key| {
                    status(key.view()).map_err(|column| {
                        tracing::debug!(
                            column,
                            "a virtual key holds what the native app does not honor"
                        );
                        Error::Forbidden
                    })
                })
                .transpose()
        })
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use chrono::DateTime;
    use rstest::rstest;
    use serde_json::value::RawValue;

    use super::*;

    fn json(text: &str) -> RawJson {
        RawJson(RawValue::from_string(text.to_owned()).unwrap())
    }

    fn at(second: i64) -> NaiveDateTime {
        DateTime::from_timestamp(second, 0).unwrap().naive_utc()
    }

    /// The row of a key generated with no parameters, as `/key/generate` stores it.
    fn plain() -> KeyView {
        KeyView {
            token: "hashed".to_owned(),
            key_name: Some("sk-...abcd".to_owned()),
            key_alias: None,
            soft_budget_cooldown: false,
            spend: 0.0,
            total_spend: 0.0,
            expires: None,
            models: vec![],
            aliases: json("{}"),
            config: json("{}"),
            router_settings: Some(json("{}")),
            user_id: None,
            team_id: None,
            agent_id: None,
            project_id: None,
            permissions: json("{}"),
            max_parallel_requests: None,
            metadata: json("{}"),
            blocked: None,
            tpm_limit: None,
            rpm_limit: None,
            tpd_limit: None,
            max_budget: None,
            soft_budget: None,
            budget_duration: None,
            budget_reset_at: None,
            allowed_cache_controls: Some(vec![]),
            allowed_routes: Some(vec![]),
            key_type: Some("default".to_owned()),
            access_group_ids: Some(vec![]),
            model_spend: json("{}"),
            model_max_budget: json("{}"),
            budget_fallbacks: json("{}"),
            budget_id: None,
            org_id: None,
            object_permission_id: None,
            created_at: Some(at(1)),
            created_by: Some("default_user_id".to_owned()),
            updated_at: Some(at(1)),
            updated_by: Some("default_user_id".to_owned()),
            settings_updated_at: None,
            last_active: None,
            rotation_count: Some(0),
            auto_rotate: Some(false),
            rotation_interval: None,
            last_rotation_at: None,
            key_rotation_at: None,
            budget_limits: None,
            team_spend: None,
            team_alias: None,
            team_tpm_limit: None,
            team_rpm_limit: None,
            team_tpd_limit: None,
            team_max_budget: None,
            team_soft_budget: None,
            team_model_max_budget: None,
            team_models: vec![],
            team_blocked: false,
            team_model_aliases: None,
            team_member: None,
            team_metadata: None,
            team_object_permission_id: None,
            team_member_spend: None,
            team_member_tpm_limit: None,
            team_member_rpm_limit: None,
            organization_alias: None,
            organization_max_budget: None,
            organization_tpm_limit: None,
            organization_rpm_limit: None,
            organization_metadata: None,
            project_alias: None,
        }
    }

    #[rstest]
    #[case::as_generated(|_: &mut KeyView| {}, None)]
    #[case::named_and_used(
        |key: &mut KeyView| {
            key.key_alias = Some("ci".to_owned());
            key.spend = 4.2;
            key.model_spend = json(r#"{"claude":4.2}"#);
            key.last_active = Some(at(50));
            key.auto_rotate = Some(true);
        },
        None
    )]
    #[case::no_key_type(|key: &mut KeyView| key.key_type = None, None)]
    #[case::unblocked(|key: &mut KeyView| key.blocked = Some(false), None)]
    #[case::expiring(|key: &mut KeyView| key.expires = Some(at(100)), Some(100))]
    fn a_key_that_holds_only_what_the_native_app_honors_is_active(
        #[case] change: fn(&mut KeyView),
        #[case] expires_at: Option<u64>,
    ) {
        let mut key = plain();
        change(&mut key);

        assert_eq!(
            status(&key),
            Ok(KeyStatus::Active {
                expires_at: expires_at
                    .map(|second| SystemTime::UNIX_EPOCH + Duration::from_secs(second)),
            })
        );
    }

    #[rstest]
    #[case::alone(|_: &mut KeyView| {})]
    #[case::of_a_team(|key: &mut KeyView| key.team_id = Some("team".to_owned()))]
    fn a_blocked_key_is_revoked(#[case] change: fn(&mut KeyView)) {
        let mut key = plain();
        key.blocked = Some(true);
        change(&mut key);

        assert_eq!(status(&key), Ok(KeyStatus::Revoked));
    }

    #[rstest]
    #[case::key_type(|key: &mut KeyView| key.key_type = Some("management".to_owned()), "key_type")]
    #[case::soft_budget_cooldown(|key: &mut KeyView| key.soft_budget_cooldown = true, "soft_budget_cooldown")]
    #[case::models(|key: &mut KeyView| key.models = vec!["claude".to_owned()], "models")]
    #[case::aliases(|key: &mut KeyView| key.aliases = json(r#"{"a":"b"}"#), "aliases")]
    #[case::config(|key: &mut KeyView| key.config = json(r#"{"a":1}"#), "config")]
    #[case::router_settings(|key: &mut KeyView| key.router_settings = Some(json(r#"{"a":1}"#)), "router_settings")]
    #[case::user_id(|key: &mut KeyView| key.user_id = Some("user".to_owned()), "user_id")]
    #[case::an_empty_user_id(|key: &mut KeyView| key.user_id = Some(String::new()), "user_id")]
    #[case::team_id(|key: &mut KeyView| key.team_id = Some("team".to_owned()), "team_id")]
    #[case::agent_id(|key: &mut KeyView| key.agent_id = Some("agent".to_owned()), "agent_id")]
    #[case::project_id(|key: &mut KeyView| key.project_id = Some("project".to_owned()), "project_id")]
    #[case::org_id(|key: &mut KeyView| key.org_id = Some("org".to_owned()), "org_id")]
    #[case::permissions(|key: &mut KeyView| key.permissions = json(r#"{"a":true}"#), "permissions")]
    #[case::max_parallel_requests(|key: &mut KeyView| key.max_parallel_requests = Some(1), "max_parallel_requests")]
    #[case::metadata(|key: &mut KeyView| key.metadata = json(r#"{"guardrails":["pii"]}"#), "metadata")]
    #[case::tpm_limit(|key: &mut KeyView| key.tpm_limit = Some(1), "tpm_limit")]
    #[case::rpm_limit(|key: &mut KeyView| key.rpm_limit = Some(1), "rpm_limit")]
    #[case::a_limit_of_zero(|key: &mut KeyView| key.rpm_limit = Some(0), "rpm_limit")]
    #[case::tpd_limit(|key: &mut KeyView| key.tpd_limit = Some(1), "tpd_limit")]
    #[case::max_budget(|key: &mut KeyView| key.max_budget = Some(5.0), "max_budget")]
    #[case::soft_budget(|key: &mut KeyView| key.soft_budget = Some(5.0), "soft_budget")]
    #[case::budget_duration(|key: &mut KeyView| key.budget_duration = Some("30d".to_owned()), "budget_duration")]
    #[case::budget_reset_at(|key: &mut KeyView| key.budget_reset_at = Some(at(9)), "budget_reset_at")]
    #[case::budget_limits(|key: &mut KeyView| key.budget_limits = Some(json(r#"[{"a":1}]"#)), "budget_limits")]
    #[case::budget_fallbacks(|key: &mut KeyView| key.budget_fallbacks = json(r#"{"a":1}"#), "budget_fallbacks")]
    #[case::budget_id(|key: &mut KeyView| key.budget_id = Some("budget".to_owned()), "budget_id")]
    #[case::model_max_budget(|key: &mut KeyView| key.model_max_budget = json(r#"{"claude":1}"#), "model_max_budget")]
    #[case::allowed_cache_controls(
        |key: &mut KeyView| key.allowed_cache_controls = Some(vec!["no-cache".to_owned()]),
        "allowed_cache_controls"
    )]
    #[case::allowed_routes(
        |key: &mut KeyView| key.allowed_routes = Some(vec!["/v1/messages".to_owned()]),
        "allowed_routes"
    )]
    #[case::access_group_ids(
        |key: &mut KeyView| key.access_group_ids = Some(vec!["group".to_owned()]),
        "access_group_ids"
    )]
    #[case::object_permission_id(
        |key: &mut KeyView| key.object_permission_id = Some("permission".to_owned()),
        "object_permission_id"
    )]
    fn a_key_that_holds_anything_else_is_named_by_the_column(
        #[case] change: fn(&mut KeyView),
        #[case] column: &str,
    ) {
        let mut key = plain();
        change(&mut key);

        assert_eq!(status(&key), Err(column));
    }

    #[rstest]
    #[case::an_object("{}")]
    #[case::a_list("[]")]
    #[case::null("null")]
    fn an_empty_json_column_holds_nothing(#[case] text: &str) {
        let key = KeyView {
            metadata: json(text),
            budget_limits: Some(json(text)),
            ..plain()
        };

        assert_eq!(carried(&key), None);
    }
}
