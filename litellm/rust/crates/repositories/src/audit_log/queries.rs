//! The insert into `LiteLLM_AuditLog`.

use serde::Serialize;
use serde_json::value::RawValue;
use sqlx::{
    PgExecutor,
    types::{Json, chrono::NaiveDateTime},
};

use litellm_db::Error;

use super::repository::NewAuditLog;
use crate::utc_timestamp::naive_utc;

/// One row as the insert reads it out of its JSON parameter.
#[derive(Serialize)]
struct Row<'a> {
    id: &'a str,
    updated_at: Option<NaiveDateTime>,
    changed_by: Option<&'a str>,
    changed_by_api_key: Option<&'a str>,
    action: &'a str,
    table_name: &'a str,
    object_id: &'a str,
    before_value: Option<&'a RawValue>,
    updated_values: Option<&'a RawValue>,
}

/// Inserts `rows` in one statement.
pub(super) async fn insert_audit_logs(
    executor: impl PgExecutor<'_>,
    rows: &[NewAuditLog],
) -> Result<(), Error> {
    let rows: Vec<Row<'_>> = rows
        .iter()
        .map(|row| Row {
            id: &row.id,
            updated_at: naive_utc(row.updated_at),
            changed_by: row.changed_by.as_deref(),
            changed_by_api_key: row.changed_by_api_key.as_deref(),
            action: &row.action,
            table_name: &row.table_name,
            object_id: &row.object_id,
            before_value: row.before_value.as_deref(),
            updated_values: row.updated_values.as_deref(),
        })
        .collect();
    sqlx::query_file!(
        "src/audit_log/queries/insert_audit_logs.sql",
        Json(rows) as _
    )
    .execute(executor)
    .await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::error::Error as StdError;

    use rstest::rstest;
    use serde_json::value::RawValue;
    use sqlx::{PgPool, Postgres, Transaction};

    use super::insert_audit_logs;
    use crate::audit_log::repository::NewAuditLog;
    use crate::test_support::{database_pool, leave_utc};

    fn row(id: &str) -> Result<NewAuditLog, Box<dyn StdError>> {
        Ok(NewAuditLog {
            id: id.to_owned(),
            updated_at: "2026-11-19T05:07:32.123456Z".parse()?,
            changed_by: Some("admin".to_owned()),
            changed_by_api_key: Some("hashed-key".to_owned()),
            action: "updated".to_owned(),
            table_name: "LiteLLM_TeamTable".to_owned(),
            object_id: "team-a".to_owned(),
            before_value: Some(RawValue::from_string(r#"{"members": []}"#.to_owned())?),
            updated_values: Some(RawValue::from_string(
                r#"{"members": ["u1"], "nested": {"n": 1}}"#.to_owned(),
            )?),
        })
    }

    async fn stored(tx: &mut Transaction<'_, Postgres>) -> Result<serde_json::Value, sqlx::Error> {
        #[expect(
            clippy::disallowed_methods,
            reason = "Assertion reads are outside the compile-checked query set"
        )]
        let rows: sqlx::types::Json<serde_json::Value> = sqlx::query_scalar(
            r#"
            SELECT COALESCE(jsonb_agg(to_jsonb(a) ORDER BY a.id), '[]')
            FROM "LiteLLM_AuditLog" AS a
            WHERE a.id LIKE 'audit-log-test-%'
            "#,
        )
        .fetch_one(&mut **tx)
        .await?;
        Ok(rows.0)
    }

    #[rstest]
    #[tokio::test]
    #[ignore = "requires DATABASE_URL pointing to a Prisma-migrated PostgreSQL database"]
    async fn audit_logs_are_stored_as_given_with_empty_defaults_for_a_missing_actor(
        #[future(awt)] database_pool: Result<PgPool, Box<dyn StdError>>,
    ) -> Result<(), Box<dyn StdError>> {
        let pool = database_pool?;
        let mut tx = pool.begin().await?;
        leave_utc(&mut tx).await?;
        let bare = NewAuditLog {
            changed_by: None,
            changed_by_api_key: None,
            action: "created".to_owned(),
            before_value: None,
            updated_values: None,
            ..row("audit-log-test-b")?
        };

        insert_audit_logs(&mut *tx, &[row("audit-log-test-a")?, bare]).await?;

        assert_eq!(
            stored(&mut tx).await?,
            serde_json::json!([
                {
                    "id": "audit-log-test-a",
                    "updated_at": "2026-11-19T05:07:32.123",
                    "changed_by": "admin",
                    "changed_by_api_key": "hashed-key",
                    "action": "updated",
                    "table_name": "LiteLLM_TeamTable",
                    "object_id": "team-a",
                    "before_value": {"members": []},
                    "updated_values": {"members": ["u1"], "nested": {"n": 1}},
                },
                {
                    "id": "audit-log-test-b",
                    "updated_at": "2026-11-19T05:07:32.123",
                    "changed_by": "",
                    "changed_by_api_key": "",
                    "action": "created",
                    "table_name": "LiteLLM_TeamTable",
                    "object_id": "team-a",
                    "before_value": null,
                    "updated_values": null,
                },
            ])
        );
        Ok(())
    }

    #[rstest]
    #[tokio::test]
    #[ignore = "requires DATABASE_URL pointing to a Prisma-migrated PostgreSQL database"]
    async fn a_batch_with_an_id_already_stored_inserts_none_of_its_rows(
        #[future(awt)] database_pool: Result<PgPool, Box<dyn StdError>>,
    ) -> Result<(), Box<dyn StdError>> {
        let pool = database_pool?;
        let mut tx = pool.begin().await?;
        insert_audit_logs(&mut *tx, &[row("audit-log-test-a")?]).await?;

        let mut attempt = sqlx::Connection::begin(&mut *tx).await?;
        let refused = insert_audit_logs(
            &mut *attempt,
            &[row("audit-log-test-new")?, row("audit-log-test-a")?],
        )
        .await;
        attempt.rollback().await?;

        let ids: Vec<String> = stored(&mut tx)
            .await?
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(|log| log["id"].as_str().map(str::to_owned))
            .collect();
        assert!(refused.is_err());
        assert_eq!(ids, ["audit-log-test-a"]);
        Ok(())
    }
}
