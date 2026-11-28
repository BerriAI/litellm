//! Budget window advances, shared with Python through the same compare-and-swap predicate.

use std::collections::BTreeMap;

use chrono::{DateTime, Utc};
use litellm_db::{Call, Error};
use serde_json::Value;

use super::queries::{self, BudgetWindows, LifetimeBudget, ResetEntity};
use crate::{Database, RawJson, Side, database::Connections};

const READ: Call = Call {
    call_type: "reset_budget_windows",
    table: None,
    operation: None,
};
const WRITE: Call = Call {
    call_type: "write_budget_windows",
    table: None,
    operation: None,
};

#[derive(Clone, Copy, Debug, strum::VariantArray)]
pub enum WindowOwner {
    Key,
    User,
    Team,
    Budget,
}

impl WindowOwner {
    fn maintained_scope(self) -> Option<&'static str> {
        match self {
            Self::Key => Some("key"),
            Self::User => Some("user"),
            Self::Team => Some("team"),
            Self::Budget => None,
        }
    }
}

pub struct WindowAdvance {
    pub duration: String,
    pub starts_at: DateTime<Utc>,
}

/// Reads rows with budget limits, and advances only the version it read.
#[derive(Clone)]
pub struct BudgetResetRepository {
    database: Database,
}

impl BudgetResetRepository {
    pub fn new(database: Database) -> Self {
        Self { database }
    }

    pub async fn advance_tier(
        &self,
        budget: &LifetimeBudget,
        end: DateTime<Utc>,
        rollover: bool,
        default_end_user_tier: Option<&str>,
    ) -> Result<Vec<ResetEntity>, Error> {
        self.database
            .run(WRITE, || async {
                let mut session = self.database.session(Side::Writer).await?;
                Ok(queries::advance_tier(
                    session.executor(),
                    budget,
                    end.naive_utc(),
                    rollover,
                    default_end_user_tier,
                )
                .await?)
            })
            .await
    }

    pub async fn lifetime_page(
        &self,
        owner: WindowOwner,
        after: &str,
        now: DateTime<Utc>,
        limit: i64,
    ) -> Result<Vec<LifetimeBudget>, Error> {
        self.database
            .run(READ, || async {
                let mut session = self.database.session(Side::Writer).await?;
                Ok(
                    queries::lifetime_page(
                        session.executor(),
                        owner,
                        after,
                        now.naive_utc(),
                        limit,
                    )
                    .await?,
                )
            })
            .await
    }

    pub async fn advance_lifetime(
        &self,
        owner: WindowOwner,
        budget: &LifetimeBudget,
        end: DateTime<Utc>,
        decrement: f64,
        credit: Option<f64>,
    ) -> Result<bool, Error> {
        self.database
            .run(WRITE, || async {
                let mut session = self.database.session(Side::Writer).await?;
                Ok(queries::advance_lifetime(
                    session.executor(),
                    owner,
                    budget,
                    end.naive_utc(),
                    decrement,
                    credit,
                )
                .await?)
            })
            .await
    }

    pub async fn window_page(
        &self,
        owner: WindowOwner,
        after: &str,
        limit: i64,
    ) -> Result<Vec<BudgetWindows>, Error> {
        self.database
            .run(READ, || async {
                let mut session = self.database.session(Side::Writer).await?;
                Ok(queries::page(session.executor(), owner, after, limit).await?)
            })
            .await
    }

    pub async fn advance_windows(
        &self,
        owner: WindowOwner,
        id: &str,
        expected: &RawJson,
        updated: &Value,
        windows: &[WindowAdvance],
    ) -> Result<bool, Error> {
        let mut distinct = BTreeMap::new();
        for window in windows {
            let start = window.starts_at.naive_utc();
            distinct
                .entry(window.duration.clone())
                .and_modify(|current| *current = start.max(*current))
                .or_insert(start);
        }
        let (durations, starts): (Vec<_>, Vec<_>) = distinct.into_iter().unzip();
        self.database
            .run(WRITE, || async {
                self.database
                    .transaction(async |transaction| {
                        advance(
                            transaction,
                            owner,
                            id,
                            expected.get(),
                            updated,
                            (&durations, &starts),
                        )
                        .await
                    })
                    .await
            })
            .await
    }
}

async fn advance(
    transaction: &mut impl Connections,
    owner: WindowOwner,
    id: &str,
    expected: &str,
    updated: &Value,
    (durations, starts): (&[String], &[chrono::NaiveDateTime]),
) -> Result<bool, Error> {
    let accepted = transaction
        .with(Side::Writer, async |executor| {
            Ok(queries::advance(executor, owner, id, expected, updated).await?)
        })
        .await?;
    if accepted && let Some(scope) = owner.maintained_scope().filter(|_| !durations.is_empty()) {
        transaction
            .with(Side::Writer, async |executor| {
                Ok(queries::roll(executor, scope, id, durations, starts).await?)
            })
            .await?;
    }
    Ok(accepted)
}

#[cfg(test)]
mod tests {
    use std::error::Error as StdError;

    use chrono::NaiveDateTime;
    use rstest::rstest;
    use serde_json::json;
    use sqlx::PgPool;

    use super::*;
    use crate::test_support::{OneTransaction, database_pool};

    #[rstest]
    #[tokio::test]
    #[ignore = "requires DATABASE_URL pointing to a Prisma-migrated PostgreSQL database"]
    async fn advancing_a_tier_once_resets_its_default_end_users_and_rolls_member_credit(
        #[future] database_pool: Result<PgPool, Box<dyn StdError>>,
    ) -> Result<(), Box<dyn StdError>> {
        let mut tx = database_pool.await?.begin().await?;
        let id = "reset-tier-cas";
        let start: NaiveDateTime = "2030-01-01T00:00:00".parse()?;
        let end: NaiveDateTime = "2030-01-02T00:00:00".parse()?;
        sqlx::query(
            r#"INSERT INTO "LiteLLM_BudgetTable"
               (budget_id, budget_duration, budget_reset_at, max_budget, rollover_max_budget, created_at, created_by, updated_by)
               VALUES ($1, '1d', $2, 10, 15, '2029-12-31T00:00:01.001', 'test', 'test')"#,
        )
        .bind(id)
        .bind(start)
        .execute(&mut *tx)
        .await?;
        sqlx::query(
            r#"INSERT INTO "LiteLLM_TeamMembership" (user_id, team_id, spend, rollover_credit, budget_id)
               VALUES ($1, 'test', 11, 3, $1)"#,
        )
        .bind(id)
        .execute(&mut *tx)
        .await?;
        sqlx::query(r#"INSERT INTO "LiteLLM_EndUserTable" (user_id, spend) VALUES ($1, 24)"#)
            .bind(id)
            .execute(&mut *tx)
            .await?;
        let python_stamp =
            "reset-tier-cas:2029-12-31T00:00:01.001000+00:00:0:2030-01-01T00:00:00+00:00";
        sqlx::query(
            r#"INSERT INTO "LiteLLM_EndUserTable" (user_id, budget_id, spend, spend_reset_window)
               VALUES ('partially-reset', $1, 14, $2)"#,
        )
        .bind(id)
        .bind(python_stamp)
        .execute(&mut *tx)
        .await?;
        let budget = LifetimeBudget {
            id: id.to_owned(),
            duration: "1d".to_owned(),
            reset_at: Some(start),
            spend: 0.0,
            max_budget: Some(10.0),
            rollover_max_budget: Some(15.0),
            rollover_credit: None,
        };
        let updates = queries::advance_tier(&mut *tx, &budget, end, true, Some(id)).await?;
        assert_eq!(updates.len(), 2);
        let stale = queries::advance_tier(&mut *tx, &budget, end, true, Some(id)).await?;
        assert!(stale.is_empty());
        let member = sqlx::query_as::<_, (f64, f64, f64)>(
            r#"SELECT spend, total_spend, rollover_credit FROM "LiteLLM_TeamMembership" WHERE user_id = $1"#,
        )
        .bind(id)
        .fetch_one(&mut *tx)
        .await?;
        let customer = sqlx::query_as::<_, (f64, String)>(
            r#"SELECT spend, spend_reset_window FROM "LiteLLM_EndUserTable" WHERE user_id = $1"#,
        )
        .bind(id)
        .fetch_one(&mut *tx)
        .await?;
        let previously_reset = sqlx::query_scalar::<_, f64>(
            r#"SELECT spend FROM "LiteLLM_EndUserTable" WHERE user_id = 'partially-reset'"#,
        )
        .fetch_one(&mut *tx)
        .await?;
        let tier = sqlx::query_as::<_, (String, NaiveDateTime, i32)>(
            r#"SELECT budget_duration, budget_reset_at, budget_reset_count FROM "LiteLLM_BudgetTable"
               WHERE budget_id = $1"#,
        )
        .bind(id)
        .fetch_one(&mut *tx)
        .await?;
        assert_eq!(member, (0.0, 11.0, 2.0));
        assert_eq!(customer, (14.0, python_stamp.to_owned()));
        assert_eq!(previously_reset, 14.0);
        assert_eq!(tier, ("1d".to_owned(), end, 1));
        Ok(())
    }

    #[rstest]
    #[tokio::test]
    #[ignore = "requires DATABASE_URL pointing to a Prisma-migrated PostgreSQL database"]
    async fn a_stale_window_advance_does_not_touch_maintained_spend(
        #[future] database_pool: Result<PgPool, Box<dyn StdError>>,
    ) -> Result<(), Box<dyn StdError>> {
        let mut tx = database_pool.await?.begin().await?;
        let id = "reset-window-cas";
        let old = json!([{"budget_duration":"1d","reset_at":"2030-01-01T00:00:00Z"}]);
        let updated = json!([{"budget_duration":"1d","reset_at":"2030-01-02T00:00:00Z"}]);
        let start: NaiveDateTime = "2030-01-01T00:00:00".parse()?;
        sqlx::query(r#"INSERT INTO "LiteLLM_UserTable" (user_id, budget_limits) VALUES ($1, $2)"#)
            .bind(id)
            .bind(&old)
            .execute(&mut *tx)
            .await?;
        sqlx::query(
            r#"INSERT INTO "LiteLLM_BudgetWindowSpend"
               (entity_type, entity_id, window_duration, window_start, spend, created_at, updated_at)
               VALUES ('user', $1, '1d', '2029-12-31', 4, NOW(), NOW())"#,
        )
        .bind(id)
        .execute(&mut *tx)
        .await?;
        let durations = ["1d".to_owned()];
        assert!(
            advance(
                &mut OneTransaction(&mut tx),
                WindowOwner::User,
                id,
                &old.to_string(),
                &updated,
                (&durations, &[start])
            )
            .await?
        );
        let future: NaiveDateTime = "2030-01-02T00:00:00".parse()?;
        assert!(
            !advance(
                &mut OneTransaction(&mut tx),
                WindowOwner::User,
                id,
                &old.to_string(),
                &old,
                (&durations, &[future])
            )
            .await?
        );
        let limits = sqlx::query_scalar::<_, Value>(
            r#"SELECT budget_limits FROM "LiteLLM_UserTable" WHERE user_id = $1"#,
        )
        .bind(id)
        .fetch_one(&mut *tx)
        .await?;
        let row = sqlx::query_as::<_, (NaiveDateTime, f64)>(
            r#"SELECT window_start, spend FROM "LiteLLM_BudgetWindowSpend" WHERE entity_type = 'user'
               AND entity_id = $1 AND window_duration = '1d'"#,
        )
        .bind(id)
        .fetch_one(&mut *tx)
        .await?;
        assert_eq!(limits, updated);
        assert_eq!(row, (start, 0.0));
        Ok(())
    }
}
