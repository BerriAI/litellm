use std::time::{Duration, SystemTime};

use serde_json::Value;
use sqlx::PgPool;

use crate::Error;

/// The columns of `LiteLLM_VerificationToken` that gate a request, with its team's, its team
/// membership's and its user's.
#[derive(Clone, Debug, PartialEq)]
pub struct VerificationToken {
    pub token: String,
    pub expires: Option<SystemTime>,
    pub blocked: bool,
    pub models: Vec<String>,
    pub spend: f64,
    pub max_budget: Option<f64>,
    pub allowed_routes: Vec<String>,
    pub key_alias: Option<String>,
    pub key_name: Option<String>,
    pub metadata: Value,
    pub restrictions: Restrictions,
    pub team_id: Option<String>,
    /// `None` when the key has no team or the team row is missing.
    pub team: Option<Team>,
    /// The budget table on the key's user's membership of the key's team; default without one.
    pub team_membership: TeamMembership,
    pub user_id: Option<String>,
    /// `None` when the key has no user or the user row is missing.
    pub user: Option<User>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Team {
    pub blocked: bool,
    pub models: Vec<String>,
    pub spend: f64,
    pub max_budget: Option<f64>,
    pub metadata: Value,
    pub restrictions: Restrictions,
}

/// What a team membership's budget table sets beyond its dollar budget.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct TeamMembership {
    /// `rpm_limit`, `tpm_limit`, `tpd_limit` or `max_parallel_requests`.
    pub rate_limits: bool,
    /// The member's own model scope; empty inherits the team's.
    pub allowed_models: Vec<String>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct User {
    pub models: Vec<String>,
    pub restrictions: Restrictions,
}

/// Whether a key, team or user sets a restriction beyond `models`, budgets, `blocked` and
/// `allowed_routes`.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct Restrictions {
    /// `rpm_limit`, `tpm_limit`, `tpd_limit` or `max_parallel_requests`, on the row or on a key's
    /// budget table, or an `rpm_limit` or `tpm_limit` inside `model_max_budget`.
    pub rate_limits: bool,
    pub policies: bool,
    /// `aliases`, `router_settings`, or a team's `model_id`.
    pub model_aliases: bool,
    pub object_permissions: bool,
    pub cache_controls: bool,
    pub project: bool,
    pub organization: bool,
    pub agent: bool,
}

/// Looks up a key by its stored token, the sha256 hex of the raw key.
pub async fn find_verification_token(
    pool: &PgPool,
    token: &str,
) -> Result<Option<VerificationToken>, Error> {
    // `expires` is a UTC `timestamp(3)`; epoch milliseconds avoid a date-time dependency.
    let row = sqlx::query!(
        r#"SELECT v.token,
                  (EXTRACT(EPOCH FROM v.expires) * 1000)::bigint AS "expires_ms?",
                  v.blocked,
                  v.models,
                  v.spend,
                  v.max_budget,
                  v.allowed_routes,
                  v.key_alias,
                  v.key_name,
                  v.metadata,
                  (v.rpm_limit IS NOT NULL OR v.tpm_limit IS NOT NULL OR v.tpd_limit IS NOT NULL
                     OR v.max_parallel_requests IS NOT NULL
                     OR kb.rpm_limit IS NOT NULL OR kb.tpm_limit IS NOT NULL
                     OR kb.tpd_limit IS NOT NULL OR kb.max_parallel_requests IS NOT NULL
                     OR COALESCE(jsonb_path_exists(v.model_max_budget,
                          '$.* ? (@.rpm_limit != null || @.tpm_limit != null)'), false)
                     OR COALESCE(jsonb_path_exists(kb.model_max_budget,
                          '$.* ? (@.rpm_limit != null || @.tpm_limit != null)'), false))
                    AS "key_rate_limits!",
                  COALESCE(cardinality(v.policies) > 0, false) AS "key_policies!",
                  (v.aliases NOT IN ('{}', 'null')
                     OR COALESCE(v.router_settings NOT IN ('{}', 'null'), false)) AS "key_model_aliases!",
                  (v.object_permission_id IS NOT NULL) AS "key_object_permissions!",
                  COALESCE(cardinality(v.allowed_cache_controls) > 0, false) AS "key_cache_controls!",
                  (v.project_id IS NOT NULL) AS "key_project!",
                  (v.organization_id IS NOT NULL) AS "key_organization!",
                  (v.agent_id IS NOT NULL) AS "key_agent!",
                  v.team_id,
                  t.team_id AS "team_row_id?",
                  t.blocked AS "team_blocked?",
                  t.models AS "team_models?",
                  t.spend AS "team_spend?",
                  t.max_budget AS "team_max_budget?",
                  t.metadata AS "team_metadata?",
                  (t.rpm_limit IS NOT NULL OR t.tpm_limit IS NOT NULL OR t.tpd_limit IS NOT NULL
                     OR t.max_parallel_requests IS NOT NULL
                     OR COALESCE(jsonb_path_exists(t.model_max_budget,
                          '$.* ? (@.rpm_limit != null || @.tpm_limit != null)'), false))
                    AS "team_rate_limits!",
                  COALESCE(cardinality(t.policies) > 0, false) AS "team_policies!",
                  (t.model_id IS NOT NULL
                     OR COALESCE(t.router_settings NOT IN ('{}', 'null'), false)) AS "team_model_aliases!",
                  (t.object_permission_id IS NOT NULL) AS "team_object_permissions!",
                  (t.organization_id IS NOT NULL) AS "team_organization!",
                  (mb.rpm_limit IS NOT NULL OR mb.tpm_limit IS NOT NULL OR mb.tpd_limit IS NOT NULL
                     OR mb.max_parallel_requests IS NOT NULL
                     OR COALESCE(jsonb_path_exists(mb.model_max_budget,
                          '$.* ? (@.rpm_limit != null || @.tpm_limit != null)'), false))
                    AS "member_rate_limits!",
                  mb.allowed_models AS "member_allowed_models?",
                  v.user_id,
                  u.user_id AS "user_row_id?",
                  u.models AS "user_models?",
                  (u.rpm_limit IS NOT NULL OR u.tpm_limit IS NOT NULL
                     OR u.max_parallel_requests IS NOT NULL
                     OR COALESCE(jsonb_path_exists(u.model_max_budget,
                          '$.* ? (@.rpm_limit != null || @.tpm_limit != null)'), false))
                    AS "user_rate_limits!",
                  COALESCE(cardinality(u.policies) > 0, false) AS "user_policies!",
                  (u.object_permission_id IS NOT NULL) AS "user_object_permissions!"
           FROM "LiteLLM_VerificationToken" AS v
           LEFT JOIN "LiteLLM_BudgetTable" AS kb ON v.budget_id = kb.budget_id
           LEFT JOIN "LiteLLM_TeamTable" AS t ON v.team_id = t.team_id
           LEFT JOIN "LiteLLM_TeamMembership" AS tm
             ON v.team_id = tm.team_id AND v.user_id = tm.user_id
           LEFT JOIN "LiteLLM_BudgetTable" AS mb ON tm.budget_id = mb.budget_id
           LEFT JOIN "LiteLLM_UserTable" AS u ON v.user_id = u.user_id
           WHERE v.token = $1"#,
        token
    )
    .fetch_optional(pool)
    .await?;
    Ok(row.map(|row| VerificationToken {
        token: row.token,
        expires: row.expires_ms.map(epoch_ms),
        blocked: row.blocked.unwrap_or(false),
        models: row.models.unwrap_or_default(),
        spend: row.spend,
        max_budget: row.max_budget,
        allowed_routes: row.allowed_routes.unwrap_or_default(),
        key_alias: row.key_alias,
        key_name: row.key_name,
        metadata: row.metadata,
        restrictions: Restrictions {
            rate_limits: row.key_rate_limits,
            policies: row.key_policies,
            model_aliases: row.key_model_aliases,
            object_permissions: row.key_object_permissions,
            cache_controls: row.key_cache_controls,
            project: row.key_project,
            organization: row.key_organization,
            agent: row.key_agent,
        },
        team_id: row.team_id,
        team: row.team_row_id.map(|_| Team {
            blocked: row.team_blocked.unwrap_or(false),
            models: row.team_models.unwrap_or_default(),
            spend: row.team_spend.unwrap_or(0.0),
            max_budget: row.team_max_budget,
            metadata: row.team_metadata.unwrap_or(Value::Null),
            restrictions: Restrictions {
                rate_limits: row.team_rate_limits,
                policies: row.team_policies,
                model_aliases: row.team_model_aliases,
                object_permissions: row.team_object_permissions,
                organization: row.team_organization,
                ..Restrictions::default()
            },
        }),
        team_membership: TeamMembership {
            rate_limits: row.member_rate_limits,
            allowed_models: row.member_allowed_models.unwrap_or_default(),
        },
        user_id: row.user_id,
        user: row.user_row_id.map(|_| User {
            models: row.user_models.unwrap_or_default(),
            restrictions: Restrictions {
                rate_limits: row.user_rate_limits,
                policies: row.user_policies,
                object_permissions: row.user_object_permissions,
                ..Restrictions::default()
            },
        }),
    }))
}

fn epoch_ms(ms: i64) -> SystemTime {
    let offset = Duration::from_millis(ms.unsigned_abs());
    if ms >= 0 {
        SystemTime::UNIX_EPOCH + offset
    } else {
        SystemTime::UNIX_EPOCH - offset
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    #[rstest]
    #[case::after_epoch(1_500, SystemTime::UNIX_EPOCH + Duration::from_millis(1_500))]
    #[case::epoch(0, SystemTime::UNIX_EPOCH)]
    #[case::before_epoch(-2_000, SystemTime::UNIX_EPOCH - Duration::from_secs(2))]
    fn converts_epoch_milliseconds(#[case] ms: i64, #[case] expected: SystemTime) {
        assert_eq!(epoch_ms(ms), expected);
    }
}
