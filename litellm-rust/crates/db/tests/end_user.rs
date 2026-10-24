use std::error::Error as StdError;

use litellm_db::{EndUser, RateLimits, find_end_user};
use rstest::{fixture, rstest};
use sqlx::{PgPool, Postgres, Transaction};

#[derive(Clone, Copy)]
struct BudgetFixture {
    budget_id: &'static str,
    max_budget: f64,
    rpm_limit: i64,
    tpm_limit: i64,
}

#[fixture]
async fn database_pool() -> Result<Option<PgPool>, sqlx::Error> {
    let Ok(database_url) = std::env::var("DATABASE_URL") else {
        return Ok(None);
    };
    PgPool::connect(&database_url).await.map(Some)
}

async fn insert_budget(
    tx: &mut Transaction<'_, Postgres>,
    budget: BudgetFixture,
) -> Result<(), sqlx::Error> {
    #[expect(
        clippy::disallowed_methods,
        reason = "Fixture inserts are outside the compile-checked query set"
    )]
    let result = sqlx::query(
        r#"
        INSERT INTO "LiteLLM_BudgetTable"
            (budget_id, max_budget, rpm_limit, tpm_limit, created_by, updated_by)
        VALUES ($1, $2, $3, $4, 'db-query-test', 'db-query-test')
        "#,
    )
    .bind(budget.budget_id)
    .bind(budget.max_budget)
    .bind(budget.rpm_limit)
    .bind(budget.tpm_limit)
    .execute(&mut **tx)
    .await;
    result.map(|_| ())
}

async fn insert_end_user(
    tx: &mut Transaction<'_, Postgres>,
    user_id: &str,
    spend: f64,
    blocked: bool,
    budget_id: Option<&str>,
) -> Result<(), sqlx::Error> {
    #[expect(
        clippy::disallowed_methods,
        reason = "Fixture inserts are outside the compile-checked query set"
    )]
    let result = sqlx::query(
        r#"
        INSERT INTO "LiteLLM_EndUserTable" (user_id, spend, blocked, budget_id)
        VALUES ($1, $2, $3, $4)
        "#,
    )
    .bind(user_id)
    .bind(spend)
    .bind(blocked)
    .bind(budget_id)
    .execute(&mut **tx)
    .await;
    result.map(|_| ())
}

#[rstest]
#[case::without_default_budget(None)]
#[case::with_default_budget(Some(BudgetFixture {
    budget_id: "missing-user-default-budget",
    max_budget: 11.5,
    rpm_limit: 12,
    tpm_limit: 1300,
}))]
#[tokio::test]
async fn missing_end_user_maps_to_defaults(
    #[case] default_budget: Option<BudgetFixture>,
    #[future(awt)] database_pool: Result<Option<PgPool>, sqlx::Error>,
) -> Result<(), Box<dyn StdError>> {
    let Some(pool) = database_pool? else {
        return Ok(());
    };
    let mut tx = pool.begin().await?;
    if let Some(budget) = default_budget {
        insert_budget(&mut tx, budget).await?;
    }

    let actual = find_end_user(
        &mut *tx,
        "missing-end-user",
        default_budget.map(|budget| budget.budget_id),
        None,
    )
    .await?;

    assert_eq!(
        actual,
        EndUser {
            spend: None,
            blocked: false,
            max_budget: default_budget.map(|budget| budget.max_budget),
            limits: RateLimits {
                rpm: default_budget.map(|budget| budget.rpm_limit),
                tpm: default_budget.map(|budget| budget.tpm_limit),
            },
        }
    );
    Ok(())
}

#[rstest]
#[case::row_budget_precedes_default(
    "row-budget-priority-user",
    Some(BudgetFixture {
        budget_id: "row-budget-priority-row",
        max_budget: 10.0,
        rpm_limit: 101,
        tpm_limit: 1001,
    }),
    BudgetFixture {
        budget_id: "row-budget-priority-default",
        max_budget: 20.0,
        rpm_limit: 201,
        tpm_limit: 2001,
    },
    None,
    Some(BudgetFixture {
        budget_id: "row-budget-priority-row",
        max_budget: 10.0,
        rpm_limit: 101,
        tpm_limit: 1001,
    }),
    false
)]
#[case::default_applies_without_row_budget(
    "default-budget-user",
    None,
    BudgetFixture {
        budget_id: "default-budget-fallback",
        max_budget: 30.0,
        rpm_limit: 301,
        tpm_limit: 3001,
    },
    None,
    Some(BudgetFixture {
        budget_id: "default-budget-fallback",
        max_budget: 30.0,
        rpm_limit: 301,
        tpm_limit: 3001,
    }),
    false
)]
#[case::override_wins_when_all_ids_differ(
    "override-priority-user",
    Some(BudgetFixture {
        budget_id: "override-priority-row",
        max_budget: 40.0,
        rpm_limit: 401,
        tpm_limit: 4001,
    }),
    BudgetFixture {
        budget_id: "override-priority-default",
        max_budget: 50.0,
        rpm_limit: 501,
        tpm_limit: 5001,
    },
    Some(BudgetFixture {
        budget_id: "override-priority-override",
        max_budget: 60.0,
        rpm_limit: 601,
        tpm_limit: 6001,
    }),
    Some(BudgetFixture {
        budget_id: "override-priority-override",
        max_budget: 60.0,
        rpm_limit: 601,
        tpm_limit: 6001,
    }),
    true
)]
#[case::dangling_override_does_not_fall_back(
    "dangling-override-user",
    Some(BudgetFixture {
        budget_id: "dangling-override-row",
        max_budget: 70.0,
        rpm_limit: 701,
        tpm_limit: 7001,
    }),
    BudgetFixture {
        budget_id: "dangling-override-default",
        max_budget: 80.0,
        rpm_limit: 801,
        tpm_limit: 8001,
    },
    Some(BudgetFixture {
        budget_id: "dangling-override-missing",
        max_budget: 90.0,
        rpm_limit: 901,
        tpm_limit: 9001,
    }),
    None,
    false
)]
#[tokio::test]
async fn budget_resolution_obeys_precedence_and_does_not_fall_back(
    #[case] user_id: &'static str,
    #[case] user_budget: Option<BudgetFixture>,
    #[case] default_budget: BudgetFixture,
    #[case] override_budget: Option<BudgetFixture>,
    #[case] expected_budget: Option<BudgetFixture>,
    #[case] insert_override_budget: bool,
    #[future(awt)] database_pool: Result<Option<PgPool>, sqlx::Error>,
) -> Result<(), Box<dyn StdError>> {
    let Some(pool) = database_pool? else {
        return Ok(());
    };
    let mut tx = pool.begin().await?;
    insert_budget(&mut tx, default_budget).await?;
    if let Some(budget) = user_budget {
        insert_budget(&mut tx, budget).await?;
    }
    if let (Some(budget), true) = (override_budget, insert_override_budget) {
        insert_budget(&mut tx, budget).await?;
    }
    insert_end_user(
        &mut tx,
        user_id,
        7.5,
        true,
        user_budget.map(|budget| budget.budget_id),
    )
    .await?;

    let actual = find_end_user(
        &mut *tx,
        user_id,
        Some(default_budget.budget_id),
        override_budget.map(|budget| budget.budget_id),
    )
    .await?;

    assert_eq!(
        actual,
        EndUser {
            spend: Some(7.5),
            blocked: true,
            max_budget: expected_budget.map(|budget| budget.max_budget),
            limits: RateLimits {
                rpm: expected_budget.map(|budget| budget.rpm_limit),
                tpm: expected_budget.map(|budget| budget.tpm_limit),
            },
        }
    );
    Ok(())
}
