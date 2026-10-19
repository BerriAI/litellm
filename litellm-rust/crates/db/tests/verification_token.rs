use std::time::{Duration, SystemTime};

use litellm_db::{
    EndUser, Restrictions, Team, TeamMembership, User, VerificationToken, find_end_user,
    find_verification_token,
};
use litellm_db_testing::MigratedPostgres;
use rstest::rstest;
use serde_json::json;

const FIXTURES: &str = r#"
INSERT INTO "LiteLLM_TeamTable" (team_id, team_alias, blocked, models, spend, max_budget, metadata,
                                 tpm_limit, model_id, created_at, updated_at)
  VALUES ('team-1', 'research', true, ARRAY['gpt-a'], 2.5, 10.0, '{"require_trace_id": true}', 100,
          NULL, now(), now()),
         ('team-2', NULL, false, ARRAY[]::text[], 0, NULL, '{}', NULL, NULL, now(), now());
INSERT INTO "LiteLLM_UserTable" (user_id, models, rpm_limit, max_budget, policies)
  VALUES ('user-1', ARRAY['gpt-b'], 5, 1.0, ARRAY['p']),
         ('user-2', ARRAY[]::text[], NULL, 2.0, ARRAY[]::text[]);
INSERT INTO "LiteLLM_BudgetTable" (budget_id, max_budget, rpm_limit, model_max_budget,
                                   allowed_models, created_by, updated_by)
  VALUES ('b-1', 5.0, NULL, NULL, ARRAY['gpt-a'], 'x', 'x'),
         ('b-2', NULL, 3, NULL, ARRAY[]::text[], 'x', 'x'),
         ('b-3', NULL, NULL, '{"gpt-a": {"tpm_limit": 7}}', ARRAY[]::text[], 'x', 'x');
INSERT INTO "LiteLLM_TeamMembership" (user_id, team_id, budget_id, spend)
  VALUES ('user-2', 'team-2', 'b-1', 0.75),
         ('user-1', 'team-1', 'b-2', 0);
INSERT INTO "LiteLLM_EndUserTable" (user_id, spend, budget_id, blocked)
  VALUES ('end-1', 1.5, 'b-1', false),
         ('end-2', 0.25, NULL, true);
INSERT INTO "LiteLLM_OrganizationTable" (organization_id, organization_alias, budget_id, created_by, updated_by)
  VALUES ('org-1', 'org', 'b-1', 'x', 'x');
INSERT INTO "LiteLLM_ProjectTable" (project_id, created_by, updated_by) VALUES ('project-1', 'x', 'x');
INSERT INTO "LiteLLM_ObjectPermissionTable" (object_permission_id) VALUES ('op-1');
INSERT INTO "LiteLLM_VerificationToken" (token, expires, blocked, models) VALUES
  ('active', NULL, NULL, ARRAY[]::text[]),
  ('blocked', NULL, true, ARRAY[]::text[]),
  ('no-models', NULL, NULL, NULL);
INSERT INTO "LiteLLM_VerificationToken"
  (token, expires, blocked, models, spend, max_budget, allowed_routes, key_alias, key_name, team_id,
   user_id, metadata)
  VALUES ('scoped', '2030-01-02 03:04:05.678', false, ARRAY['gpt-a', 'claude-*'], 1.5, 3.0,
          ARRAY['llm_api_routes'], 'prod', 'sk-...abcd', 'team-1', 'user-1', '{"tags": ["a"]}'),
         ('orphan', NULL, false, ARRAY[]::text[], 0, NULL, ARRAY[]::text[], NULL, NULL, 'gone',
          NULL, '{}'),
         ('member', NULL, false, ARRAY[]::text[], 0, NULL, ARRAY[]::text[], NULL, NULL, 'team-2',
          'user-2', '{}');
INSERT INTO "LiteLLM_VerificationToken"
  (token, models, rpm_limit, budget_id, policies, aliases, allowed_cache_controls, project_id,
   organization_id, agent_id, object_permission_id)
  VALUES ('restricted', ARRAY[]::text[], 10, 'b-1', ARRAY['p'], '{"a": "b"}', ARRAY['no-cache'],
          'project-1', 'org-1', 'agent-1', 'op-1');
INSERT INTO "LiteLLM_VerificationToken" (token, models, max_parallel_requests)
  VALUES ('limited-only', ARRAY[]::text[], 2);
INSERT INTO "LiteLLM_VerificationToken"
  (token, models, max_budget, budget_id, budget_limits, model_max_budget)
  VALUES ('budgeted', ARRAY[]::text[], 4.0, 'b-1', '[{"budget_duration": "1d", "max_budget": 1}]',
          '{"gpt-a": {"budget_limit": 1.0, "time_period": "1d", "rpm_limit": null}}'),
         ('budget-row-limits', ARRAY[]::text[], NULL, 'b-2', NULL, '{}'),
         ('budget-row-model-limits', ARRAY[]::text[], NULL, 'b-3', NULL, '{}'),
         ('model-rpm', ARRAY[]::text[], NULL, NULL, NULL,
          '{"gpt-a": {"budget_limit": 1.0, "rpm_limit": 2}}');
"#;

#[expect(
    clippy::disallowed_methods,
    reason = "test fixture rows; the lookups under test use the checked macro"
)]
async fn setup(pool: &sqlx::PgPool) {
    sqlx::raw_sql(FIXTURES).execute(pool).await.unwrap();
}

#[rstest]
#[tokio::test]
async fn reads_the_gating_columns_of_a_verification_token_and_its_owners() {
    let postgres = MigratedPostgres::start().await.unwrap();
    let pool = postgres.pool();
    setup(pool).await;
    let find = |token: &'static str| async move {
        find_verification_token(pool, token).await.unwrap().unwrap()
    };

    assert_eq!(
        find("active").await,
        VerificationToken {
            token: "active".into(),
            expires: None,
            blocked: false,
            models: vec![],
            spend: 0.0,
            max_budget: None,
            allowed_routes: vec![],
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
    );
    assert_eq!(
        find("scoped").await,
        VerificationToken {
            token: "scoped".into(),
            // 2030-01-02T03:04:05.678Z
            expires: Some(SystemTime::UNIX_EPOCH + Duration::from_millis(1_893_553_445_678)),
            blocked: false,
            models: vec!["gpt-a".into(), "claude-*".into()],
            spend: 1.5,
            max_budget: Some(3.0),
            allowed_routes: vec!["llm_api_routes".into()],
            key_alias: Some("prod".into()),
            key_name: Some("sk-...abcd".into()),
            metadata: json!({"tags": ["a"]}),
            restrictions: Restrictions::default(),
            team_id: Some("team-1".into()),
            team: Some(Team {
                alias: Some("research".into()),
                blocked: true,
                models: vec!["gpt-a".into()],
                spend: 2.5,
                max_budget: Some(10.0),
                metadata: json!({"require_trace_id": true}),
                restrictions: Restrictions {
                    rate_limits: true,
                    ..Restrictions::default()
                },
            }),
            team_membership: TeamMembership {
                spend: Some(0.0),
                max_budget: None,
                default_max_budget: None,
                rate_limits: true,
                allowed_models: vec![],
            },
            user_id: Some("user-1".into()),
            user: Some(User {
                spend: 0.0,
                max_budget: Some(1.0),
                models: vec!["gpt-b".into()],
                restrictions: Restrictions {
                    rate_limits: true,
                    policies: true,
                    ..Restrictions::default()
                },
            }),
        }
    );
    assert_eq!(
        find("restricted").await.restrictions,
        Restrictions {
            rate_limits: true,
            policies: true,
            model_aliases: true,
            object_permissions: true,
            cache_controls: true,
            project: true,
            organization: true,
            agent: true,
        }
    );
    assert_eq!(
        find("limited-only").await.restrictions,
        Restrictions {
            rate_limits: true,
            ..Restrictions::default()
        }
    );
    // Budgets are not restrictions; limits held in a budget row or per model are.
    assert_eq!(find("budgeted").await.restrictions, Restrictions::default());
    for token in ["budget-row-limits", "budget-row-model-limits", "model-rpm"] {
        assert_eq!(
            find(token).await.restrictions,
            Restrictions {
                rate_limits: true,
                ..Restrictions::default()
            },
            "{token}"
        );
    }
    let member = find("member").await;
    assert_eq!(
        member.team_membership,
        TeamMembership {
            spend: Some(0.75),
            max_budget: Some(5.0),
            default_max_budget: None,
            rate_limits: false,
            allowed_models: vec!["gpt-a".into()],
        }
    );
    // A key without its own budget takes its budget row's.
    assert_eq!(find("restricted").await.max_budget, Some(5.0));
    assert_eq!(find("budgeted").await.max_budget, Some(4.0));
    assert_eq!(member.team.unwrap().restrictions, Restrictions::default());
    assert_eq!(member.user.unwrap().restrictions, Restrictions::default());
    let orphan = find("orphan").await;
    assert_eq!(
        (orphan.team_id.as_deref(), orphan.team),
        (Some("gone"), None)
    );
    assert!(find("blocked").await.blocked);
    assert_eq!(find("no-models").await.models, Vec::<String>::new());
    assert_eq!(
        find_verification_token(pool, "missing").await.unwrap(),
        None
    );
}

#[rstest]
#[tokio::test]
async fn reads_end_user_budgets_with_their_default() {
    let postgres = MigratedPostgres::start().await.unwrap();
    let pool = postgres.pool();
    setup(pool).await;
    let find =
        async |id: &str, default: Option<&str>| find_end_user(pool, id, default).await.unwrap();

    assert_eq!(
        find("end-1", Some("b-2")).await,
        EndUser {
            spend: Some(1.5),
            blocked: false,
            max_budget: Some(5.0)
        }
    );
    assert_eq!(
        find("end-2", Some("b-1")).await,
        EndUser {
            spend: Some(0.25),
            blocked: true,
            max_budget: Some(5.0)
        }
    );
    assert_eq!(find("end-2", None).await.max_budget, None);
    assert_eq!(
        find("unknown", Some("b-1")).await,
        EndUser {
            spend: None,
            blocked: false,
            max_budget: Some(5.0)
        }
    );
}
