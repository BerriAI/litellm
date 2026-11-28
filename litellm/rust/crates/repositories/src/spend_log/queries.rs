//! Sums of `LiteLLM_SpendLogs.spend` for one key, team or user since a window start
//!
//! The spend log ledger's statements on `LiteLLM_SpendLogs` and `LiteLLM_SpendLogDelivery` sit here too,
//! mirroring the Python ledger writer (`SpendLogsRepository._insert_ledger`).

use chrono::NaiveDateTime;
use serde::Serialize;
use serde_json::{Value, value::RawValue};
use sqlx::{PgExecutor, types::Json};

use litellm_db::Error;

use super::SpendLogOwner;
use super::accounting::rollups::DailyTable;
use super::accounting::{Entity, WindowCharge, WindowKey};

pub(super) async fn set_accounting_lock_timeout(
    executor: impl PgExecutor<'_>,
    setting: &str,
) -> Result<(), Error> {
    sqlx::query_file!(
        "src/spend_log/queries/set_accounting_lock_timeout.sql",
        setting
    )
    .fetch_one(executor)
    .await?;
    Ok(())
}

pub(super) async fn apply_daily_spend(
    executor: impl PgExecutor<'_>,
    table: DailyTable,
    rows: &Value,
) -> Result<(), Error> {
    macro_rules! apply {
        ($file:literal) => {
            sqlx::query_file!($file, rows).execute(executor).await?
        };
    }
    match table {
        DailyTable::Agent => apply!("src/spend_log/queries/apply_accounting_daily_agent.sql"),
        DailyTable::CustomerMember => {
            apply!("src/spend_log/queries/apply_accounting_daily_customer_member.sql")
        }
        DailyTable::EndUser => apply!("src/spend_log/queries/apply_accounting_daily_end_user.sql"),
        DailyTable::Organization => {
            apply!("src/spend_log/queries/apply_accounting_daily_organization.sql")
        }
        DailyTable::Tag => apply!("src/spend_log/queries/apply_accounting_daily_tag.sql"),
        DailyTable::Team => apply!("src/spend_log/queries/apply_accounting_daily_team.sql"),
        DailyTable::User => apply!("src/spend_log/queries/apply_accounting_daily_user.sql"),
    };
    Ok(())
}

pub(super) async fn apply_model_usage(
    executor: impl PgExecutor<'_>,
    rows: &Value,
) -> Result<(), Error> {
    sqlx::query_file!(
        "src/spend_log/queries/apply_accounting_model_usage.sql",
        rows
    )
    .execute(executor)
    .await?;
    Ok(())
}

pub(super) async fn apply_guardrail_usage(
    executor: impl PgExecutor<'_>,
    rows: &Value,
) -> Result<(), Error> {
    sqlx::query_file!(
        "src/spend_log/queries/apply_accounting_guardrail_usage.sql",
        rows
    )
    .execute(executor)
    .await?;
    Ok(())
}

pub(super) async fn lock_teams(executor: impl PgExecutor<'_>, ids: &[String]) -> Result<(), Error> {
    sqlx::query_file!("src/spend_log/queries/lock_accounting_teams.sql", ids)
        .fetch_all(executor)
        .await?;
    Ok(())
}

pub(super) async fn update_entities(
    executor: impl PgExecutor<'_>,
    entity: Entity,
    charges: &Value,
) -> Result<(), Error> {
    macro_rules! apply {
        ($file:literal) => {
            sqlx::query_file!($file, charges).execute(executor).await?
        };
    }
    match entity {
        Entity::Organization => apply!("src/spend_log/queries/apply_accounting_organization.sql"),
        Entity::OrganizationMember => {
            apply!("src/spend_log/queries/apply_accounting_organization_member.sql")
        }
        Entity::Team => apply!("src/spend_log/queries/apply_accounting_team.sql"),
        Entity::User => apply!("src/spend_log/queries/apply_accounting_user.sql"),
        Entity::EndUser => apply!("src/spend_log/queries/apply_accounting_end_user.sql"),
        Entity::TeamMember => apply!("src/spend_log/queries/apply_accounting_team_member.sql"),
        Entity::Key => apply!("src/spend_log/queries/apply_accounting_key.sql"),
        Entity::Project => apply!("src/spend_log/queries/apply_accounting_project.sql"),
        Entity::Tag => apply!("src/spend_log/queries/apply_accounting_tag.sql"),
        Entity::ModelAccessGroup => {
            apply!("src/spend_log/queries/apply_accounting_model_access_group.sql")
        }
        Entity::Agent => apply!("src/spend_log/queries/apply_accounting_agent.sql"),
        Entity::AppGroup => apply!("src/spend_log/queries/apply_accounting_app_group.sql"),
        Entity::AppGroupCustomerOrg => {
            apply!("src/spend_log/queries/apply_accounting_app_group_customer_org.sql")
        }
        Entity::App => apply!("src/spend_log/queries/apply_accounting_app.sql"),
        Entity::AppCustomerOrg => {
            apply!("src/spend_log/queries/apply_accounting_app_customer_org.sql")
        }
        Entity::AppCustomerMember => {
            apply!("src/spend_log/queries/apply_accounting_app_customer_member.sql")
        }
        Entity::CustomerOrg => apply!("src/spend_log/queries/apply_accounting_customer_org.sql"),
        Entity::CustomerTeam => apply!("src/spend_log/queries/apply_accounting_customer_team.sql"),
        Entity::CustomerMember => {
            apply!("src/spend_log/queries/apply_accounting_customer_member.sql")
        }
    };
    Ok(())
}

pub(super) async fn window_seed_base(
    executor: impl PgExecutor<'_>,
    key: &WindowKey<'_>,
    charge: &WindowCharge,
) -> Result<f64, Error> {
    Ok(sqlx::query_file_scalar!(
        "src/spend_log/queries/seed_accounting_window.sql",
        key.entity_type,
        key.entity_id,
        key.duration,
        key.starts_at,
        charge.earliest_request,
        charge.spend,
    )
    .fetch_one(executor)
    .await?)
}

pub(super) async fn apply_window(
    executor: impl PgExecutor<'_>,
    key: &WindowKey<'_>,
    charge: &WindowCharge,
    seed: f64,
) -> Result<(), Error> {
    let initial = seed + charge.spend;
    sqlx::query_file!(
        "src/spend_log/queries/apply_accounting_window.sql",
        key.entity_type,
        key.entity_id,
        key.duration,
        key.starts_at,
        initial,
        charge.spend,
    )
    .execute(executor)
    .await?;
    Ok(())
}

pub(super) struct ClaimedDelivery {
    pub(super) delivery_id: String,
    pub(super) attempts: i32,
    pub(super) accounting_payload: Value,
}

#[derive(Serialize)]
pub(super) struct DeliveryAttempt {
    pub(super) delivery_id: String,
    pub(super) attempts: i32,
}

pub(super) async fn claim_deliveries(
    executor: impl PgExecutor<'_>,
    limit: i32,
    retry: bool,
) -> Result<Vec<ClaimedDelivery>, Error> {
    Ok(sqlx::query_file_as!(
        ClaimedDelivery,
        "src/spend_log/queries/claim_accounting_deliveries.sql",
        limit,
        retry,
    )
    .fetch_all(executor)
    .await?)
}

pub(super) async fn mark_applied(
    executor: impl PgExecutor<'_>,
    ids: &[String],
) -> Result<(), Error> {
    sqlx::query_file!(
        "src/spend_log/queries/mark_accounting_deliveries_applied.sql",
        ids
    )
    .execute(executor)
    .await?;
    Ok(())
}

pub(super) async fn park_undecodable(
    executor: impl PgExecutor<'_>,
    ids: &[String],
) -> Result<(), Error> {
    sqlx::query_file!(
        "src/spend_log/queries/park_undecodable_accounting_deliveries.sql",
        ids
    )
    .execute(executor)
    .await?;
    Ok(())
}

pub(super) async fn record_retry(
    executor: impl PgExecutor<'_>,
    claimed: &Value,
    error: &str,
    park: bool,
) -> Result<usize, Error> {
    let updated = sqlx::query_file!(
        "src/spend_log/queries/retry_accounting_deliveries.sql",
        claimed,
        error,
        park
    )
    .execute(executor)
    .await?
    .rows_affected();
    Ok(usize::try_from(updated).unwrap_or(usize::MAX))
}

pub(super) async fn current_window_spend(
    executor: impl PgExecutor<'_>,
    entity_type: &str,
    id: &str,
    duration: &str,
    expected_start: NaiveDateTime,
) -> Result<Option<f64>, Error> {
    Ok(sqlx::query_file_scalar!(
        "src/spend_log/queries/find_current_window_spend.sql",
        entity_type,
        id,
        duration,
        expected_start,
    )
    .fetch_optional(executor)
    .await?)
}

pub(super) async fn cleanup_accounting_deliveries(
    executor: impl PgExecutor<'_>,
    days: i32,
    limit: i32,
) -> Result<u64, Error> {
    Ok(sqlx::query_file!(
        "src/spend_log/queries/cleanup_accounting_deliveries.sql",
        days,
        limit
    )
    .execute(executor)
    .await?
    .rows_affected())
}

macro_rules! window_spend_sql {
    ($column:literal) => {
        concat!(
            r#"SELECT COALESCE(SUM(spend), 0) FROM "LiteLLM_SpendLogs" WHERE "#,
            $column,
            r#" = $1 AND "startTime" >= $2"#
        )
    };
}

pub(super) async fn window_spend(
    executor: impl PgExecutor<'_>,
    owner: SpendLogOwner,
    owner_id: &str,
    since: NaiveDateTime,
) -> Result<f64, Error> {
    let sql = match owner {
        SpendLogOwner::Key => window_spend_sql!("api_key"),
        SpendLogOwner::Team => window_spend_sql!("team_id"),
        SpendLogOwner::User => window_spend_sql!(r#""user""#),
    };
    #[expect(
        clippy::disallowed_methods,
        reason = "The column comes from matching SpendLogOwner, so the SQL is one of three static strings"
    )]
    let query = sqlx::query_scalar::<_, f64>(sql);
    Ok(query.bind(owner_id).bind(since).fetch_one(executor).await?)
}

/// Inserts the request logs in `records` (a JSON array of `LiteLLM_SpendLogs` rows), skipping
/// any `request_id` already stored, and returns the ids it inserted.
pub(super) async fn insert_spend_log_rows(
    executor: impl PgExecutor<'_>,
    records: Json<Box<RawValue>>,
) -> Result<Vec<String>, Error> {
    Ok(sqlx::query_file_scalar!(
        "src/spend_log/queries/insert_spend_log_rows.sql",
        records as _
    )
    .fetch_all(executor)
    .await?)
}

/// Inserts the deliveries in `payloads` (a JSON array of delivery rows), skipping any
/// `delivery_id` already stored, and returns the ids it accepted.
pub(super) async fn insert_spend_log_deliveries(
    executor: impl PgExecutor<'_>,
    payloads: Value,
) -> Result<Vec<String>, Error> {
    Ok(sqlx::query_file_scalar!(
        "src/spend_log/queries/insert_spend_log_deliveries.sql",
        payloads
    )
    .fetch_all(executor)
    .await?)
}

/// Which of the settle-once `request_ids` a pre-ledger cost row already charged.
pub(super) async fn spend_log_jobs_charged_by_legacy(
    executor: impl PgExecutor<'_>,
    request_ids: &[String],
) -> Result<Vec<String>, Error> {
    Ok(sqlx::query_file_scalar!(
        "src/spend_log/queries/select_spend_log_jobs_charged_by_legacy.sql",
        request_ids
    )
    .fetch_all(executor)
    .await?)
}

/// Removes the deliveries in `delivery_ids`.
pub(super) async fn delete_spend_log_deliveries(
    executor: impl PgExecutor<'_>,
    delivery_ids: &[String],
) -> Result<(), Error> {
    sqlx::query_file!(
        "src/spend_log/queries/delete_spend_log_deliveries.sql",
        delivery_ids
    )
    .execute(executor)
    .await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::error::Error as StdError;

    use chrono::NaiveDateTime;
    use rstest::rstest;
    use serde_json::json;
    use sqlx::{PgPool, Postgres, Transaction};

    use super::{
        DeliveryAttempt, SpendLogOwner, claim_deliveries, mark_applied, park_undecodable,
        record_retry, window_spend,
    };
    use crate::test_support::database_pool;

    /// One log owned by `owner` in the column under test, with `other` in the other two owner columns. The column
    /// goes into the request id so the cases, which run at once, never wait on each other's primary key.
    async fn insert_spend_log(
        tx: &mut Transaction<'_, Postgres>,
        column: SpendLogOwner,
        request_id: &str,
        owner: &str,
        other: &str,
        spend: f64,
        start_time: &str,
    ) -> Result<(), sqlx::Error> {
        let (api_key, team_id, user) = match column {
            SpendLogOwner::Key => (owner, other, other),
            SpendLogOwner::Team => (other, owner, other),
            SpendLogOwner::User => (other, other, owner),
        };
        #[expect(
            clippy::disallowed_methods,
            reason = "Fixture inserts are outside the compile-checked query set"
        )]
        let result = sqlx::query(
            r#"
            INSERT INTO "LiteLLM_SpendLogs"
                (request_id, call_type, api_key, team_id, "user", spend, "startTime", "endTime")
            VALUES ($1, 'acompletion', $2, $3, $4, $5, $6::timestamp, $6::timestamp)
            "#,
        )
        .bind(format!("{request_id}-{column:?}"))
        .bind(api_key)
        .bind(team_id)
        .bind(user)
        .bind(spend)
        .bind(start_time)
        .execute(&mut **tx)
        .await;
        result.map(|_| ())
    }

    fn window_start() -> Result<NaiveDateTime, chrono::ParseError> {
        NaiveDateTime::parse_from_str("2026-08-01 00:00:00", "%Y-%m-%d %H:%M:%S")
    }

    #[rstest]
    #[case::key(SpendLogOwner::Key)]
    #[case::team(SpendLogOwner::Team)]
    #[case::user(SpendLogOwner::User)]
    #[tokio::test]
    #[ignore = "requires DATABASE_URL pointing to a Prisma-migrated PostgreSQL database"]
    async fn sums_only_the_owners_column_from_the_window_start_on(
        #[future(awt)] database_pool: Result<PgPool, Box<dyn StdError>>,
        #[case] column: SpendLogOwner,
    ) -> Result<(), Box<dyn StdError>> {
        let pool = database_pool?;
        let mut tx = pool.begin().await?;
        for (request_id, owner, other, spend, start_time) in [
            (
                "window-before",
                "window-owner",
                "window-x",
                100.0,
                "2026-07-31 23:59:59.999",
            ),
            (
                "window-at-start",
                "window-owner",
                "window-x",
                1.25,
                "2026-08-01 00:00:00",
            ),
            (
                "window-inside",
                "window-owner",
                "window-x",
                2.5,
                "2026-08-03 12:00:00",
            ),
            (
                "window-other-column",
                "window-x",
                "window-owner",
                50.0,
                "2026-08-02 00:00:00",
            ),
        ] {
            insert_spend_log(&mut tx, column, request_id, owner, other, spend, start_time).await?;
        }

        assert_eq!(
            window_spend(&mut *tx, column, "window-owner", window_start()?).await?,
            3.75
        );
        Ok(())
    }

    #[rstest]
    #[case::key(SpendLogOwner::Key)]
    #[case::team(SpendLogOwner::Team)]
    #[case::user(SpendLogOwner::User)]
    #[tokio::test]
    #[ignore = "requires DATABASE_URL pointing to a Prisma-migrated PostgreSQL database"]
    async fn no_rows_in_the_window_is_zero(
        #[future(awt)] database_pool: Result<PgPool, Box<dyn StdError>>,
        #[case] column: SpendLogOwner,
    ) -> Result<(), Box<dyn StdError>> {
        let pool = database_pool?;
        let mut tx = pool.begin().await?;
        insert_spend_log(
            &mut tx,
            column,
            "window-old",
            "window-quiet-owner",
            "window-x",
            9.0,
            "2026-07-01 00:00:00",
        )
        .await?;
        let since = window_start()?;

        assert_eq!(
            window_spend(&mut *tx, column, "window-quiet-owner", since).await?,
            0.0
        );
        assert_eq!(
            window_spend(&mut *tx, column, "window-missing-owner", since).await?,
            0.0
        );
        Ok(())
    }

    #[rstest]
    #[case::applied("applied", 0, None)]
    #[case::undecodable("parked", 1, Some("UndecodablePayload"))]
    #[case::retry("pending", 1, Some("DataRejected"))]
    #[case::attempts_exhausted("parked", 1, Some("DataRejected"))]
    #[tokio::test]
    #[ignore = "requires DATABASE_URL pointing to a Prisma-migrated PostgreSQL database"]
    async fn a_claimed_delivery_gets_its_terminal_state_or_backoff(
        #[future(awt)] database_pool: Result<PgPool, Box<dyn StdError>>,
        #[case] expected_state: &str,
        #[case] expected_attempts: i32,
        #[case] expected_error: Option<&str>,
    ) -> Result<(), Box<dyn StdError>> {
        let pool = database_pool?;
        let mut tx = pool.begin().await?;
        let id = format!("accounting-claim-{}", uuid::Uuid::new_v4());
        #[expect(
            clippy::disallowed_methods,
            reason = "Fixture inserts are outside the compile-checked query set"
        )]
        sqlx::query(
            r#"INSERT INTO "LiteLLM_SpendLogDelivery"
                (delivery_id, request_id, accounting_payload, is_batch, next_attempt_at)
                VALUES ($1, $1, '{}', false, '1900-01-01')"#,
        )
        .bind(&id)
        .execute(&mut *tx)
        .await?;

        let claims = claim_deliveries(&mut *tx, 1, false).await?;
        assert_eq!(claims.len(), 1);
        assert_eq!(claims[0].delivery_id, id);
        assert_eq!(claims[0].accounting_payload, json!({}));
        match expected_error {
            None => mark_applied(&mut *tx, std::slice::from_ref(&id)).await?,
            Some("UndecodablePayload") => {
                park_undecodable(&mut *tx, std::slice::from_ref(&id)).await?;
            }
            Some(error) => {
                let attempt = DeliveryAttempt {
                    delivery_id: id.clone(),
                    attempts: claims[0].attempts,
                };
                record_retry(
                    &mut *tx,
                    &json!([attempt]),
                    error,
                    expected_state == "parked",
                )
                .await?;
            }
        }

        #[expect(
            clippy::disallowed_methods,
            reason = "Fixture assertions are outside the compile-checked query set"
        )]
        let state = sqlx::query_as::<_, (String, i32, Option<String>, bool, bool)>(
            r#"SELECT state::text, attempts, last_error, applied_at IS NOT NULL,
                   next_attempt_at > (NOW() AT TIME ZONE 'UTC')
               FROM "LiteLLM_SpendLogDelivery" WHERE delivery_id = $1"#,
        )
        .bind(&id)
        .fetch_one(&mut *tx)
        .await?;
        assert_eq!(
            state,
            (
                expected_state.to_owned(),
                expected_attempts,
                expected_error.map(str::to_owned),
                expected_state == "applied",
                expected_error == Some("DataRejected"),
            )
        );
        Ok(())
    }
}
