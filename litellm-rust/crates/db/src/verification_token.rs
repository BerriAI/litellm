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
    /// The key's `rpm_limit` and `tpm_limit`, else its budget table's.
    pub limits: RateLimits,
    pub max_parallel_requests: Option<i64>,
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
    pub alias: Option<String>,
    pub blocked: bool,
    pub models: Vec<String>,
    pub spend: f64,
    pub max_budget: Option<f64>,
    pub metadata: Value,
    pub limits: RateLimits,
    pub restrictions: Restrictions,
}

/// Requests and tokens per rate limit window.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct RateLimits {
    pub rpm: Option<i64>,
    pub tpm: Option<i64>,
}

/// The key's team membership: its spend and budget, and what its budget table sets beyond them.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct TeamMembership {
    /// The membership row's spend, when the key's user has a row for its team.
    pub spend: Option<f64>,
    /// The membership's own budget, with any temporary increase still in effect.
    pub max_budget: Option<f64>,
    /// The team's default member budget, `metadata.team_member_budget_id`.
    pub default_max_budget: Option<f64>,
    /// The membership budget table's `rpm_limit` and `tpm_limit`.
    pub limits: RateLimits,
    /// An `rpm_limit` or `tpm_limit` inside the budget table's `model_max_budget`.
    pub rate_limits: bool,
    /// The member's own model scope; empty inherits the team's.
    pub allowed_models: Vec<String>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct User {
    pub spend: f64,
    pub max_budget: Option<f64>,
    pub models: Vec<String>,
    pub limits: RateLimits,
    pub restrictions: Restrictions,
}

/// Whether a key, team or user sets a restriction beyond `models`, budgets, `blocked` and
/// `allowed_routes`.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct Restrictions {
    /// Per-model limits: an `rpm_limit` or `tpm_limit` inside `model_max_budget`, on the row or
    /// on a key's budget table.
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
                  COALESCE(v.max_budget, kb.max_budget) AS max_budget,
                  v.allowed_routes,
                  v.key_alias,
                  v.key_name,
                  v.metadata,
                  COALESCE(v.rpm_limit, kb.rpm_limit) AS rpm_limit,
                  COALESCE(v.tpm_limit, kb.tpm_limit) AS tpm_limit,
                  v.max_parallel_requests,
                  (COALESCE(jsonb_path_exists(v.model_max_budget,
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
                  t.team_alias AS "team_alias?",
                  t.blocked AS "team_blocked?",
                  t.models AS "team_models?",
                  t.spend AS "team_spend?",
                  t.max_budget AS "team_max_budget?",
                  t.metadata AS "team_metadata?",
                  t.rpm_limit AS "team_rpm_limit?",
                  t.tpm_limit AS "team_tpm_limit?",
                  (COALESCE(jsonb_path_exists(t.model_max_budget,
                          '$.* ? (@.rpm_limit != null || @.tpm_limit != null)'), false))
                    AS "team_rate_limits!",
                  COALESCE(cardinality(t.policies) > 0, false) AS "team_policies!",
                  (t.model_id IS NOT NULL
                     OR COALESCE(t.router_settings NOT IN ('{}', 'null'), false)) AS "team_model_aliases!",
                  (t.object_permission_id IS NOT NULL) AS "team_object_permissions!",
                  (t.organization_id IS NOT NULL) AS "team_organization!",
                  mb.rpm_limit AS "member_rpm_limit?",
                  mb.tpm_limit AS "member_tpm_limit?",
                  (COALESCE(jsonb_path_exists(mb.model_max_budget,
                          '$.* ? (@.rpm_limit != null || @.tpm_limit != null)'), false))
                    AS "member_rate_limits!",
                  mb.allowed_models AS "member_allowed_models?",
                  tm.spend AS "member_spend?",
                  mb.max_budget AS "member_max_budget?",
                  CASE WHEN mb.temp_budget_expiry > (now() AT TIME ZONE 'UTC')
                       THEN mb.temp_budget_increase END AS "member_budget_increase?",
                  tdb.max_budget AS "member_default_budget?",
                  v.user_id,
                  u.user_id AS "user_row_id?",
                  u.models AS "user_models?",
                  u.spend AS "user_spend?",
                  u.max_budget AS "user_max_budget?",
                  u.rpm_limit AS "user_rpm_limit?",
                  u.tpm_limit AS "user_tpm_limit?",
                  (COALESCE(jsonb_path_exists(u.model_max_budget,
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
           LEFT JOIN "LiteLLM_BudgetTable" AS tdb
             ON tdb.budget_id = t.metadata ->> 'team_member_budget_id'
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
        limits: RateLimits {
            rpm: row.rpm_limit,
            tpm: row.tpm_limit,
        },
        max_parallel_requests: row.max_parallel_requests.map(i64::from),
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
            alias: row.team_alias,
            blocked: row.team_blocked.unwrap_or(false),
            models: row.team_models.unwrap_or_default(),
            spend: row.team_spend.unwrap_or(0.0),
            max_budget: row.team_max_budget,
            metadata: row.team_metadata.unwrap_or(Value::Null),
            limits: RateLimits {
                rpm: row.team_rpm_limit,
                tpm: row.team_tpm_limit,
            },
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
            spend: row.member_spend,
            max_budget: row
                .member_max_budget
                .map(|budget| budget + row.member_budget_increase.unwrap_or(0.0)),
            default_max_budget: row.member_default_budget,
            limits: RateLimits {
                rpm: row.member_rpm_limit,
                tpm: row.member_tpm_limit,
            },
            rate_limits: row.member_rate_limits,
            allowed_models: row.member_allowed_models.unwrap_or_default(),
        },
        user_id: row.user_id,
        user: row.user_row_id.map(|_| User {
            spend: row.user_spend.unwrap_or(0.0),
            max_budget: row.user_max_budget,
            models: row.user_models.unwrap_or_default(),
            limits: RateLimits {
                rpm: row.user_rpm_limit,
                tpm: row.user_tpm_limit,
            },
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
