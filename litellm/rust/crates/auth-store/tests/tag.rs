//! Tag reads against migrated Postgres, using the same cache entries as the Redis and Python wire tests.

mod support;

use std::{collections::BTreeMap, sync::Arc};

use litellm_auth_store::{Error, TagDatabase};
use litellm_db_testing::MigratedPostgres;
use litellm_repositories::TagRepository;
use rstest::{fixture, rstest};

use support::{database, dumped, execute, insert_budgets};

const PLAIN_ENTRY: &str = concat!(
    r#"{"tag_name": "plain", "description": null, "models": [], "model_info": null,"#,
    r#" "spend": 0.0, "budget_id": null, "litellm_budget_table": null,"#,
    r#" "created_at": "2026-11-19T10:00:00Z", "created_by": null, "updated_at": "2026-11-19T10:00:00Z"}"#,
);
const LIMITED_ENTRY: &str = concat!(
    r#"{"tag_name": "limited", "description": "Team tag", "models": ["claude"],"#,
    r#" "model_info": {"deployment-1": "claude"}, "spend": 1.5, "budget_id": "bare","#,
    r#" "litellm_budget_table": {"budget_id": "bare", "soft_budget": null, "max_budget": 10.5,"#,
    r#" "max_parallel_requests": null, "tpm_limit": 1000, "rpm_limit": 3, "tpd_limit": null,"#,
    r#" "model_max_budget": null, "model_tpm_limit": null, "model_rpm_limit": null, "budget_duration": null,"#,
    r#" "allowed_models": [], "temp_budget_increase": null, "temp_budget_expiry": null},"#,
    r#" "created_at": "2026-11-19T10:00:00.123000Z", "created_by": "creator","#,
    r#" "updated_at": "2026-11-19T10:00:00Z"}"#,
);

#[fixture]
async fn tags() -> (MigratedPostgres, TagDatabase) {
    let postgres = MigratedPostgres::start().await.unwrap();
    let pool = postgres.pool();
    insert_budgets(pool).await;
    execute(
        pool,
        r#"UPDATE "LiteLLM_BudgetTable" SET max_budget = 10.5, rpm_limit = 3, tpm_limit = 1000
           WHERE budget_id = 'bare'"#,
    )
    .await;
    execute(
        pool,
        r#"INSERT INTO "LiteLLM_TagTable" (
               tag_name, description, models, model_info, spend, budget_id, created_at, created_by, updated_at
           ) VALUES
               ('plain', NULL, NULL, NULL, 0, NULL, '2026-11-19 10:00:00', NULL, '2026-11-19 10:00:00'),
               ('limited', 'Team tag', ARRAY['claude'], '{"deployment-1":"claude"}', 1.5, 'bare',
                '2026-11-19 10:00:00.123', 'creator', '2026-11-19 10:00:00'),
               ('malformed', NULL, NULL, '[]', 0, NULL, '2026-11-19 10:00:00', NULL, '2026-11-19 10:00:00')"#,
    )
    .await;
    let tags = TagDatabase::new(Arc::new(TagRepository::new(database(&postgres))));
    (postgres, tags)
}

#[rstest]
#[tokio::test(flavor = "multi_thread")]
async fn a_registry_read_returns_at_most_the_requested_number_of_names(
    #[future(awt)] tags: (MigratedPostgres, TagDatabase),
) {
    let (_postgres, tags) = tags;
    let mut all = tags.names(4).await.unwrap();
    all.sort_unstable();

    assert_eq!(all, ["limited", "malformed", "plain"]);
    let first = tags.names(1).await.unwrap();
    assert!(matches!(first.as_slice(), [name] if all.contains(name)));
}

#[rstest]
#[tokio::test(flavor = "multi_thread")]
async fn a_tag_batch_returns_only_the_requested_rows_that_exist_in_the_shape_python_caches(
    #[future(awt)] tags: (MigratedPostgres, TagDatabase),
) {
    let (_postgres, tags) = tags;
    let rows = tags
        .tags(&[
            "plain".to_owned(),
            "missing".to_owned(),
            "limited".to_owned(),
        ])
        .await
        .unwrap();

    assert_eq!(
        rows.into_iter()
            .map(|entry| (entry.tag_name.clone(), dumped(entry)))
            .collect::<BTreeMap<_, _>>(),
        BTreeMap::from([
            ("limited".to_owned(), LIMITED_ENTRY.to_owned()),
            ("plain".to_owned(), PLAIN_ENTRY.to_owned()),
        ])
    );
}

#[rstest]
#[tokio::test(flavor = "multi_thread")]
async fn a_tag_batch_with_an_invalid_model_info_column_fails_as_a_whole(
    #[future(awt)] tags: (MigratedPostgres, TagDatabase),
) {
    let (_postgres, tags) = tags;
    let error = tags
        .tags(&["plain".to_owned(), "malformed".to_owned()])
        .await
        .unwrap_err();

    assert!(matches!(
        error,
        Error::Row {
            column: "model_info",
            within: None
        }
    ));
}
