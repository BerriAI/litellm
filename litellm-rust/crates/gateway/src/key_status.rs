use litellm_db::{Restrictions, VerificationToken};
use litellm_gateway_auth::keys::{
    AllowedRoutes, KeyModels, KeyStatus, ModelAccess, Owner, Refusal, Restriction,
};
use serde_json::Value;

/// Key and team `metadata` fields the proxy enforces and this gateway does not.
const METADATA_RESTRICTIONS: [(&str, Restriction); 11] = [
    ("guardrails", Restriction::Guardrails),
    ("policies", Restriction::Guardrails),
    ("tags", Restriction::Tags),
    ("model_rpm_limit", Restriction::RateLimits),
    ("model_tpm_limit", Restriction::RateLimits),
    ("tag_rpm_limit", Restriction::RateLimits),
    ("allowed_tools", Restriction::ToolAllowlists),
    ("enforced_params", Restriction::EnforcedParams),
    ("priority", Restriction::Priority),
    ("max_iterations", Restriction::MaxIterations),
    ("require_trace_id", Restriction::RequiredTraceIds),
];

/// Refuses what the proxy would refuse, and every key carrying a restriction the gateway does not
/// enforce yet.
pub(crate) fn status(row: VerificationToken, personal_keys_use_teams: bool) -> KeyStatus {
    if row.blocked {
        return KeyStatus::Blocked;
    }
    if let Some(refusal) = refusal(&row, personal_keys_use_teams) {
        return KeyStatus::Refused {
            expires_at: row.expires,
            refusal,
        };
    }
    KeyStatus::Active {
        expires_at: row.expires,
        models: KeyModels {
            key: ModelAccess::new(row.models),
            team: row.team.map(|team| ModelAccess::new(team.models)),
        },
        allowed_routes: AllowedRoutes::new(row.allowed_routes),
        team_id: row.team_id,
    }
}

fn refusal(row: &VerificationToken, personal_keys_use_teams: bool) -> Option<Refusal> {
    let team = match (&row.team_id, &row.team) {
        (Some(team_id), None) => {
            return Some(Refusal::TeamMissing {
                team_id: team_id.clone(),
            });
        }
        (Some(team_id), Some(team)) => Some((team_id, team)),
        (None, _) => None,
    };
    if let Some((team_id, team)) = team
        && team.blocked
    {
        return Some(Refusal::TeamBlocked {
            team_id: team_id.clone(),
        });
    }
    // Budgets pass: the gateway does not record spend for its own traffic, so it refuses only a
    // key or team whose stored spend has already reached its budget.
    if let Some(max_budget) = row.max_budget
        && max_budget.is_finite()
        && row.spend >= max_budget
    {
        return Some(budget_exceeded(
            format!("Key={}", key_label(row)),
            row.spend,
            max_budget,
        ));
    }
    if let Some((team_id, team)) = team
        && let Some(max_budget) = team.max_budget
        && max_budget.is_finite()
        && team.spend > max_budget
    {
        return Some(budget_exceeded(
            format!("Team={team_id}"),
            team.spend,
            max_budget,
        ));
    }
    unsupported(row, personal_keys_use_teams)
        .map(|(restriction, owner)| Refusal::Unsupported { restriction, owner })
}

fn unsupported(
    row: &VerificationToken,
    personal_keys_use_teams: bool,
) -> Option<(Restriction, Owner)> {
    let key = restriction(&row.restrictions).or_else(|| metadata_restriction(&row.metadata));
    let team = || {
        row.team.as_ref().and_then(|team| {
            restriction(&team.restrictions).or_else(|| metadata_restriction(&team.metadata))
        })
    };
    let user = || {
        row.user.as_ref().and_then(|user| {
            restriction(&user.restrictions).or_else(|| {
                (row.team_id.is_none() && limits_models(&user.models))
                    .then_some(Restriction::ModelLists)
            })
        })
    };
    key.map(|restriction| (restriction, Owner::Key))
        .or_else(|| team().map(|restriction| (restriction, Owner::Team)))
        .or_else(|| membership(row).map(|restriction| (restriction, Owner::TeamMembership)))
        .or_else(|| user().map(|restriction| (restriction, Owner::User)))
        .or_else(|| {
            (personal_keys_use_teams && row.team_id.is_none() && row.user_id.is_some())
                .then_some((Restriction::PersonalKeyTeamModels, Owner::Proxy))
        })
}

/// What the membership's budget table sets: rate limits, or an `allowed_models` list the proxy's
/// `check_team_member_model_access` can deny with (it skips an empty list, and `*` and
/// `all-proxy-models` grant every model).
fn membership(row: &VerificationToken) -> Option<Restriction> {
    let membership = &row.team_membership;
    if membership.rate_limits {
        return Some(Restriction::RateLimits);
    }
    let models = &membership.allowed_models;
    let listed = |name: &str| models.iter().any(|model| model == name);
    (!(models.is_empty() || listed("*") || listed("all-proxy-models")))
        .then_some(Restriction::ModelLists)
}

/// Whether the proxy's `can_user_call_model` can deny a model for this `models` list: anything
/// but an empty list or one granting all models, and always with `no-default-models`.
fn limits_models(models: &[String]) -> bool {
    let listed = |name: &str| models.iter().any(|model| model == name);
    listed("no-default-models") || !(models.is_empty() || listed("*") || listed("all-proxy-models"))
}

fn restriction(restrictions: &Restrictions) -> Option<Restriction> {
    [
        (restrictions.rate_limits, Restriction::RateLimits),
        (restrictions.policies, Restriction::Guardrails),
        (restrictions.model_aliases, Restriction::ModelAliases),
        (
            restrictions.object_permissions,
            Restriction::ObjectPermissions,
        ),
        (restrictions.cache_controls, Restriction::CacheControls),
        (restrictions.project, Restriction::Project),
        (restrictions.organization, Restriction::Organization),
        (restrictions.agent, Restriction::Agent),
    ]
    .into_iter()
    .find_map(|(set, restriction)| set.then_some(restriction))
}

fn metadata_restriction(metadata: &Value) -> Option<Restriction> {
    if let Value::String(text) = metadata {
        return serde_json::from_str::<Value>(text)
            .ok()
            .filter(Value::is_object)
            .and_then(|parsed| metadata_restriction(&parsed));
    }
    METADATA_RESTRICTIONS
        .iter()
        .find(|(field, _)| metadata.get(field).is_some_and(is_set))
        .map(|(_, restriction)| *restriction)
}

fn is_set(value: &Value) -> bool {
    match value {
        Value::Null | Value::Bool(false) => false,
        Value::String(text) => !text.is_empty(),
        Value::Array(items) => !items.is_empty(),
        Value::Object(fields) => !fields.is_empty(),
        Value::Bool(true) | Value::Number(_) => true,
    }
}

fn budget_exceeded(label: String, spend: f64, max_budget: f64) -> Refusal {
    Refusal::BudgetExceeded {
        owner: label,
        spend: python_float(spend),
        max_budget: python_float(max_budget),
    }
}

/// The proxy names a key by its alias, adding `key_name` only while it still looks masked.
fn key_label(row: &VerificationToken) -> String {
    let alias = row
        .key_alias
        .as_deref()
        .filter(|alias| !alias.is_empty())
        .unwrap_or("key");
    match row.key_name.as_deref().filter(|name| is_masked(name)) {
        Some(name) => format!("{alias} ({name})"),
        None => alias.to_owned(),
    }
}

/// The proxy's `_MASKED_KEY_NAME_RE`, `sk-\.\.\.(?:[^\s\x00-\x1f\x7f-\x9f]{4})?`.
fn is_masked(name: &str) -> bool {
    name.strip_prefix("sk-...").is_some_and(|rest| {
        rest.is_empty()
            || (rest.chars().count() == 4
                && rest
                    .chars()
                    .all(|character| !character.is_whitespace() && !character.is_control()))
    })
}

/// Python's `repr` of a float.
fn python_float(value: f64) -> String {
    if value.is_nan() {
        return "nan".into();
    }
    if value.is_infinite() {
        return if value > 0.0 { "inf" } else { "-inf" }.into();
    }
    let magnitude = value.abs();
    if magnitude != 0.0 && !(1e-4..1e16).contains(&magnitude) {
        let scientific = format!("{value:e}");
        let (mantissa, exponent) = scientific.split_once('e').unwrap_or((&scientific, "0"));
        let exponent: i32 = exponent.parse().unwrap_or(0);
        let sign = if exponent < 0 { '-' } else { '+' };
        return format!("{mantissa}e{sign}{:02}", exponent.abs());
    }
    format!("{value:?}")
}

#[cfg(test)]
mod tests {
    use std::time::SystemTime;

    use litellm_db::{Team, TeamMembership, User};
    use rstest::{fixture, rstest};
    use serde_json::json;

    use super::*;

    type Edit = fn(VerificationToken) -> VerificationToken;

    fn strings(values: &[&str]) -> Vec<String> {
        values.iter().map(|value| (*value).to_owned()).collect()
    }

    #[fixture]
    fn row() -> VerificationToken {
        VerificationToken {
            token: "hash".into(),
            expires: Some(SystemTime::UNIX_EPOCH),
            blocked: false,
            models: strings(&["gpt-a"]),
            spend: 0.0,
            max_budget: None,
            allowed_routes: strings(&["llm_api_routes"]),
            key_alias: None,
            key_name: None,
            metadata: json!({}),
            restrictions: Restrictions::default(),
            team_id: None,
            team: None,
            team_membership: TeamMembership::default(),
            user_id: None,
            user: None,
        }
    }

    fn team(blocked: bool, max_budget: Option<f64>, spend: f64) -> Option<Team> {
        Some(Team {
            blocked,
            models: strings(&["gpt-b"]),
            spend,
            max_budget,
            metadata: json!({}),
            restrictions: Restrictions::default(),
        })
    }

    fn active(team: bool) -> KeyStatus {
        KeyStatus::Active {
            expires_at: Some(SystemTime::UNIX_EPOCH),
            models: KeyModels {
                key: ModelAccess::new(strings(&["gpt-a"])),
                team: team.then(|| ModelAccess::new(strings(&["gpt-b"]))),
            },
            allowed_routes: AllowedRoutes::new(strings(&["llm_api_routes"])),
            team_id: team.then(|| "team-1".to_owned()),
        }
    }

    fn refused(refusal: Refusal) -> KeyStatus {
        KeyStatus::Refused {
            expires_at: Some(SystemTime::UNIX_EPOCH),
            refusal,
        }
    }

    fn exceeded(owner: &str, spend: &str, max_budget: &str) -> KeyStatus {
        refused(Refusal::BudgetExceeded {
            owner: owner.into(),
            spend: spend.into(),
            max_budget: max_budget.into(),
        })
    }

    fn unsupported(restriction: Restriction, owner: Owner) -> KeyStatus {
        refused(Refusal::Unsupported { restriction, owner })
    }

    #[rstest]
    fn maps_an_active_row_with_its_team_and_routes(row: VerificationToken) {
        let row = VerificationToken {
            team_id: Some("team-1".into()),
            team: team(false, None, 0.0),
            ..row
        };
        assert_eq!(status(row, false), active(true));
    }

    #[rstest]
    #[case::blocked_key(true, None, None, None, 0.0, KeyStatus::Blocked)]
    #[case::team_row_missing(
        false,
        Some("team-1"),
        None,
        None,
        0.0,
        refused(Refusal::TeamMissing { team_id: "team-1".into() })
    )]
    #[case::team_blocked(
        false,
        Some("team-1"),
        team(true, None, 0.0),
        None,
        0.0,
        refused(Refusal::TeamBlocked { team_id: "team-1".into() })
    )]
    #[case::key_at_budget(
        false,
        None,
        None,
        Some(10.0),
        10.0,
        exceeded("Key=key", "10.0", "10.0")
    )]
    #[case::key_under_budget(false, None, None, Some(10.0), 9.5, active(false))]
    #[case::key_zero_budget(false, None, None, Some(0.0), 0.0, exceeded("Key=key", "0.0", "0.0"))]
    #[case::key_infinite_budget(false, None, None, Some(f64::INFINITY), 1e20, active(false))]
    #[case::team_over_budget(
        false,
        Some("team-1"),
        team(false, Some(5.0), 5.25),
        None,
        0.0,
        exceeded("Team=team-1", "5.25", "5.0")
    )]
    #[case::team_at_budget(
        false,
        Some("team-1"),
        team(false, Some(5.0), 5.0),
        None,
        0.0,
        active(true)
    )]
    #[case::team_over_budget_under_key_budget(
        false,
        Some("team-1"),
        team(false, Some(5.0), 9.0),
        Some(10.0),
        2.0,
        exceeded("Team=team-1", "9.0", "5.0")
    )]
    #[case::key_and_team_under_budget(
        false,
        Some("team-1"),
        team(false, Some(5.0), 1.0),
        Some(10.0),
        2.0,
        active(true)
    )]
    #[case::key_budget_checked_before_team(
        false,
        Some("team-1"),
        team(false, Some(5.0), 9.0),
        Some(1.0),
        2.0,
        exceeded("Key=key", "2.0", "1.0")
    )]
    fn refuses_like_the_proxy(
        row: VerificationToken,
        #[case] blocked: bool,
        #[case] team_id: Option<&str>,
        #[case] team: Option<Team>,
        #[case] max_budget: Option<f64>,
        #[case] spend: f64,
        #[case] expected: KeyStatus,
    ) {
        let row = VerificationToken {
            blocked,
            team_id: team_id.map(str::to_owned),
            team,
            max_budget,
            spend,
            ..row
        };
        assert_eq!(status(row, false), expected);
    }

    fn key(restrictions: Restrictions, row: VerificationToken) -> VerificationToken {
        VerificationToken {
            restrictions,
            ..row
        }
    }

    fn key_metadata(metadata: Value, row: VerificationToken) -> VerificationToken {
        VerificationToken { metadata, ..row }
    }

    fn on_team(
        restrictions: Restrictions,
        metadata: Value,
        row: VerificationToken,
    ) -> VerificationToken {
        VerificationToken {
            team_id: Some("team-1".into()),
            team: team(false, None, 0.0).map(|team| Team {
                restrictions,
                metadata,
                ..team
            }),
            ..row
        }
    }

    fn on_user(
        team_id: Option<&str>,
        models: &[&str],
        restrictions: Restrictions,
        row: VerificationToken,
    ) -> VerificationToken {
        VerificationToken {
            team_id: team_id.map(str::to_owned),
            team: team_id.and_then(|_| team(false, None, 0.0)),
            user_id: Some("user-1".into()),
            user: Some(User {
                models: strings(models),
                restrictions,
            }),
            ..row
        }
    }

    fn on_membership(
        rate_limits: bool,
        allowed_models: &[&str],
        row: VerificationToken,
    ) -> VerificationToken {
        VerificationToken {
            team_membership: TeamMembership {
                rate_limits,
                allowed_models: strings(allowed_models),
            },
            ..on_team(NONE, json!({}), row)
        }
    }

    const RATE_LIMITS: Restrictions = Restrictions {
        rate_limits: true,
        ..NONE
    };
    const POLICIES: Restrictions = Restrictions {
        policies: true,
        ..NONE
    };
    const MODEL_ALIASES: Restrictions = Restrictions {
        model_aliases: true,
        ..NONE
    };
    const OBJECT_PERMISSIONS: Restrictions = Restrictions {
        object_permissions: true,
        ..NONE
    };
    const CACHE_CONTROLS: Restrictions = Restrictions {
        cache_controls: true,
        ..NONE
    };
    const PROJECT: Restrictions = Restrictions {
        project: true,
        ..NONE
    };
    const ORGANIZATION: Restrictions = Restrictions {
        organization: true,
        ..NONE
    };
    const AGENT: Restrictions = Restrictions {
        agent: true,
        ..NONE
    };
    const NONE: Restrictions = Restrictions {
        rate_limits: false,
        policies: false,
        model_aliases: false,
        object_permissions: false,
        cache_controls: false,
        project: false,
        organization: false,
        agent: false,
    };

    #[rstest]
    #[case::key_rate_limits(|row| key(RATE_LIMITS, row), Some((Restriction::RateLimits, Owner::Key)))]
    #[case::key_policies(|row| key(POLICIES, row), Some((Restriction::Guardrails, Owner::Key)))]
    #[case::key_model_aliases(|row| key(MODEL_ALIASES, row), Some((Restriction::ModelAliases, Owner::Key)))]
    #[case::key_object_permissions(
        |row| key(OBJECT_PERMISSIONS, row),
        Some((Restriction::ObjectPermissions, Owner::Key))
    )]
    #[case::key_cache_controls(|row| key(CACHE_CONTROLS, row), Some((Restriction::CacheControls, Owner::Key)))]
    #[case::key_project(|row| key(PROJECT, row), Some((Restriction::Project, Owner::Key)))]
    #[case::key_organization(|row| key(ORGANIZATION, row), Some((Restriction::Organization, Owner::Key)))]
    #[case::key_agent(|row| key(AGENT, row), Some((Restriction::Agent, Owner::Key)))]
    #[case::metadata_guardrails(
        |row| key_metadata(json!({"guardrails": ["pii"]}), row),
        Some((Restriction::Guardrails, Owner::Key))
    )]
    #[case::metadata_policies(
        |row| key_metadata(json!({"policies": ["p"]}), row),
        Some((Restriction::Guardrails, Owner::Key))
    )]
    #[case::metadata_tags(|row| key_metadata(json!({"tags": ["a"]}), row), Some((Restriction::Tags, Owner::Key)))]
    #[case::metadata_model_rpm(
        |row| key_metadata(json!({"model_rpm_limit": {"gpt-a": 1}}), row),
        Some((Restriction::RateLimits, Owner::Key))
    )]
    #[case::metadata_model_tpm(
        |row| key_metadata(json!({"model_tpm_limit": {"gpt-a": 1}}), row),
        Some((Restriction::RateLimits, Owner::Key))
    )]
    #[case::metadata_tag_rpm(
        |row| key_metadata(json!({"tag_rpm_limit": {"a": 1}}), row),
        Some((Restriction::RateLimits, Owner::Key))
    )]
    #[case::metadata_allowed_tools(
        |row| key_metadata(json!({"allowed_tools": ["search"]}), row),
        Some((Restriction::ToolAllowlists, Owner::Key))
    )]
    #[case::metadata_enforced_params(
        |row| key_metadata(json!({"enforced_params": ["user"]}), row),
        Some((Restriction::EnforcedParams, Owner::Key))
    )]
    #[case::metadata_priority(
        |row| key_metadata(json!({"priority": "high"}), row),
        Some((Restriction::Priority, Owner::Key))
    )]
    #[case::metadata_max_iterations(
        |row| key_metadata(json!({"max_iterations": 0}), row),
        Some((Restriction::MaxIterations, Owner::Key))
    )]
    #[case::metadata_trace_id(
        |row| key_metadata(json!({"require_trace_id": true}), row),
        Some((Restriction::RequiredTraceIds, Owner::Key))
    )]
    #[case::metadata_as_json_string(
        |row| key_metadata(json!("{\"tags\": [\"a\"]}"), row),
        Some((Restriction::Tags, Owner::Key))
    )]
    #[case::metadata_unset_values(
        |row| key_metadata(
            json!({"tags": [], "guardrails": null, "require_trace_id": false, "priority": ""}),
            row
        ),
        None
    )]
    #[case::metadata_other_fields(
        |row| key_metadata(json!({"team": "x", "spend_logs_metadata": {"a": 1}}), row),
        None
    )]
    #[case::team_rate_limits(
        |row| on_team(RATE_LIMITS, json!({}), row),
        Some((Restriction::RateLimits, Owner::Team))
    )]
    #[case::team_policies(|row| on_team(POLICIES, json!({}), row), Some((Restriction::Guardrails, Owner::Team)))]
    #[case::team_model_aliases(
        |row| on_team(MODEL_ALIASES, json!({}), row),
        Some((Restriction::ModelAliases, Owner::Team))
    )]
    #[case::team_object_permissions(
        |row| on_team(OBJECT_PERMISSIONS, json!({}), row),
        Some((Restriction::ObjectPermissions, Owner::Team))
    )]
    #[case::team_organization(
        |row| on_team(ORGANIZATION, json!({}), row),
        Some((Restriction::Organization, Owner::Team))
    )]
    #[case::team_metadata_trace_id(
        |row| on_team(NONE, json!({"require_trace_id": true}), row),
        Some((Restriction::RequiredTraceIds, Owner::Team))
    )]
    #[case::team_metadata_model_rpm(
        |row| on_team(NONE, json!({"model_rpm_limit": {"gpt-a": 1}}), row),
        Some((Restriction::RateLimits, Owner::Team))
    )]
    #[case::plain_team(|row| on_team(NONE, json!({}), row), None)]
    #[case::membership_budget_only(|row| on_membership(false, &[], row), None)]
    #[case::membership_rate_limits(
        |row| on_membership(true, &[], row),
        Some((Restriction::RateLimits, Owner::TeamMembership))
    )]
    #[case::membership_allowed_models(
        |row| on_membership(false, &["gpt-b"], row),
        Some((Restriction::ModelLists, Owner::TeamMembership))
    )]
    #[case::membership_star(|row| on_membership(false, &["gpt-b", "*"], row), None)]
    #[case::membership_all_proxy_models(|row| on_membership(false, &["all-proxy-models"], row), None)]
    #[case::user_rate_limits(
        |row| on_user(None, &[], RATE_LIMITS, row),
        Some((Restriction::RateLimits, Owner::User))
    )]
    #[case::user_policies(|row| on_user(None, &[], POLICIES, row), Some((Restriction::Guardrails, Owner::User)))]
    #[case::user_object_permissions(
        |row| on_user(None, &[], OBJECT_PERMISSIONS, row),
        Some((Restriction::ObjectPermissions, Owner::User))
    )]
    #[case::user_models_on_personal_key(
        |row| on_user(None, &["gpt-a"], NONE, row),
        Some((Restriction::ModelLists, Owner::User))
    )]
    #[case::user_no_default_models(
        |row| on_user(None, &["*", "no-default-models"], NONE, row),
        Some((Restriction::ModelLists, Owner::User))
    )]
    #[case::user_star(|row| on_user(None, &["gpt-a", "*"], NONE, row), None)]
    #[case::user_all_proxy_models(|row| on_user(None, &["all-proxy-models"], NONE, row), None)]
    #[case::user_models_ignored_on_team_key(|row| on_user(Some("team-1"), &["gpt-a"], NONE, row), None)]
    #[case::plain_user(|row| on_user(None, &[], NONE, row), None)]
    fn refuses_restrictions_the_gateway_does_not_enforce(
        row: VerificationToken,
        #[case] edit: Edit,
        #[case] expected: Option<(Restriction, Owner)>,
    ) {
        let status = status(edit(row), false);
        match expected {
            Some((restriction, owner)) => assert_eq!(status, unsupported(restriction, owner)),
            None => assert!(matches!(status, KeyStatus::Active { .. }), "{status:?}"),
        }
    }

    #[rstest]
    #[case::personal_key(None, Some("user-1"))]
    #[case::team_key(Some("team-1"), Some("user-1"))]
    #[case::no_user(None, None)]
    fn refuses_personal_keys_when_teams_grant_their_models(
        row: VerificationToken,
        #[case] team_id: Option<&str>,
        #[case] user_id: Option<&str>,
    ) {
        let row = VerificationToken {
            team_id: team_id.map(str::to_owned),
            team: team_id.and_then(|_| team(false, None, 0.0)),
            user_id: user_id.map(str::to_owned),
            ..row
        };
        let personal = team_id.is_none() && user_id.is_some();
        assert_eq!(status(row.clone(), false), active(team_id.is_some()));
        assert_eq!(
            status(row, true),
            if personal {
                unsupported(Restriction::PersonalKeyTeamModels, Owner::Proxy)
            } else {
                active(team_id.is_some())
            }
        );
    }

    #[rstest]
    #[case::no_alias(None, None, "key")]
    #[case::empty_alias(Some(""), None, "key")]
    #[case::alias(Some("prod"), None, "prod")]
    #[case::masked_name(Some("prod"), Some("sk-...abcd"), "prod (sk-...abcd)")]
    #[case::bare_masked_name(None, Some("sk-..."), "key (sk-...)")]
    #[case::unmasked_name(Some("prod"), Some("sk-unmasked90"), "prod")]
    #[case::name_with_space(Some("prod"), Some("sk-...ab d"), "prod")]
    #[case::name_too_long(Some("prod"), Some("sk-...abcde"), "prod")]
    fn labels_keys_like_the_proxy(
        row: VerificationToken,
        #[case] alias: Option<&str>,
        #[case] name: Option<&str>,
        #[case] expected: &str,
    ) {
        let row = VerificationToken {
            key_alias: alias.map(str::to_owned),
            key_name: name.map(str::to_owned),
            ..row
        };
        assert_eq!(key_label(&row), expected);
    }

    #[rstest]
    #[case::whole(10.0, "10.0")]
    #[case::fraction(0.1, "0.1")]
    #[case::zero(0.0, "0.0")]
    #[case::small(0.0001, "0.0001")]
    #[case::tiny(0.00001, "1e-05")]
    #[case::large(1e15, "1000000000000000.0")]
    #[case::huge(1e16, "1e+16")]
    #[case::huge_fraction(1.5e17, "1.5e+17")]
    #[case::negative(-2.5, "-2.5")]
    #[case::infinite(f64::INFINITY, "inf")]
    fn prints_floats_like_python(#[case] value: f64, #[case] expected: &str) {
        assert_eq!(python_float(value), expected);
    }
}
