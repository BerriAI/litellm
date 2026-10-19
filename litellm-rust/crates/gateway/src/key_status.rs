use litellm_db::{Restrictions, VerificationToken};
use std::sync::Arc;

use litellm_gateway_auth::keys::{
    AllowedRoutes, KeyBudgets, KeyModels, KeyProfile, KeyStatus, MemberSpend, ModelAccess, Owner,
    Refusal, Restriction, Spend,
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
    let object = |value: Value| match value {
        Value::Object(fields) => Arc::new(fields),
        _ => Arc::default(),
    };
    let budgets = budgets(&row);
    let (team_models, team_alias, team_metadata) = match row.team {
        Some(team) => (
            Some(ModelAccess::new(team.models)),
            team.alias,
            object(team.metadata),
        ),
        None => (None, None, Arc::default()),
    };
    KeyStatus::Active {
        expires_at: row.expires,
        models: KeyModels {
            key: ModelAccess::new(row.models),
            team: team_models,
        },
        allowed_routes: AllowedRoutes::new(row.allowed_routes),
        team_id: row.team_id,
        profile: Arc::new(KeyProfile {
            alias: row.key_alias,
            user_id: row.user_id,
            team_alias,
            metadata: object(row.metadata),
            team_metadata,
            budgets,
        }),
    }
}

/// The spend and budgets the budgets hook checks for this key.
fn budgets(row: &VerificationToken) -> KeyBudgets {
    let membership = &row.team_membership;
    KeyBudgets {
        key: Spend {
            spend: row.spend,
            max_budget: row.max_budget,
        },
        key_name: row.key_name.clone(),
        team: row.team.as_ref().map(|team| Spend {
            spend: team.spend,
            max_budget: team.max_budget,
        }),
        user: row.user.as_ref().map(|user| Spend {
            spend: user.spend,
            max_budget: user.max_budget,
        }),
        team_member: membership.spend.map(|spend| MemberSpend {
            spend,
            max_budget: membership.max_budget,
            default_max_budget: membership.default_max_budget,
        }),
        end_user_budget_id: row
            .metadata
            .get("end_user_budget_id")
            .and_then(Value::as_str)
            .map(str::to_owned),
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
            alias: None,
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
            profile: Arc::new(KeyProfile {
                budgets: KeyBudgets {
                    team: team.then(Spend::default),
                    ..KeyBudgets::default()
                },
                ..KeyProfile::default()
            }),
        }
    }

    fn refused(refusal: Refusal) -> KeyStatus {
        KeyStatus::Refused {
            expires_at: Some(SystemTime::UNIX_EPOCH),
            refusal,
        }
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
                spend: 0.0,
                max_budget: None,
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
                ..TeamMembership::default()
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
        let expected = match active(team_id.is_some()) {
            KeyStatus::Active {
                expires_at,
                models,
                allowed_routes,
                team_id,
                profile,
            } => KeyStatus::Active {
                expires_at,
                models,
                allowed_routes,
                team_id,
                profile: Arc::new(KeyProfile {
                    user_id: user_id.map(str::to_owned),
                    ..(*profile).clone()
                }),
            },
            other => other,
        };
        assert_eq!(status(row.clone(), false), expected);
        assert_eq!(
            status(row, true),
            if personal {
                unsupported(Restriction::PersonalKeyTeamModels, Owner::Proxy)
            } else {
                expected
            }
        );
    }
}
