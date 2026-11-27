//! Reads every object a key's auth checks read in one round trip, each as the JSON object its cache entry holds.

use sqlx::PgExecutor;

use litellm_db::Error;

use crate::RawJson;

/// The user, team, team membership, organization and project rows, each `None` when it was not asked for or is
/// missing. Each row carries the budget, object permission and model alias rows auth reads with it.
#[macro_rules_attribute::apply(crate::db_row)]
#[derive(serde::Serialize)]
pub struct KeyOwners {
    user_row: Option<RawJson>,
    team_row: Option<RawJson>,
    membership_row: Option<RawJson>,
    organization_row: Option<RawJson>,
    project_row: Option<RawJson>,
}

pub(super) async fn find_key_owners(
    executor: impl PgExecutor<'_>,
    user_id: Option<&str>,
    team_id: Option<&str>,
    membership_user_id: Option<&str>,
    organization_id: Option<&str>,
    project_id: Option<&str>,
) -> Result<KeyOwners, Error> {
    Ok(sqlx::query_file_as!(
        KeyOwners,
        "src/key_owners/queries/find_key_owners.sql",
        user_id,
        team_id,
        membership_user_id,
        organization_id,
        project_id,
    )
    .fetch_one(executor)
    .await?)
}

#[cfg(test)]
mod tests {
    use std::error::Error as StdError;

    use rstest::rstest;
    use sqlx::{PgPool, Postgres, Transaction};

    use super::find_key_owners;
    use crate::test_support::{database_pool, insert_budget};

    async fn insert_owners(tx: &mut Transaction<'_, Postgres>) -> Result<(), sqlx::Error> {
        insert_budget(tx, "owners-budget").await?;
        #[expect(
            clippy::disallowed_methods,
            reason = "Fixture inserts are outside the compile-checked query set"
        )]
        let result = sqlx::query(
            r#"
            WITH users AS (
                INSERT INTO "LiteLLM_UserTable" (user_id, models, metadata, created_at, updated_at)
                VALUES ('owners-user', NULL, '{}', '2026-11-01 00:00:00', '2026-11-01 00:00:00')
            ),
            teams AS (
                INSERT INTO "LiteLLM_TeamTable" (team_id, admins, members, metadata, created_at, updated_at)
                VALUES ('owners-team', '{}', '{}', '{}', '2026-11-01 00:00:00', '2026-11-01 00:00:00')
            )
            INSERT INTO "LiteLLM_ProjectTable"
                (project_id, budget_id, metadata, models, spend, model_spend, blocked,
                 created_at, created_by, updated_at, updated_by)
            VALUES ('owners-project', 'owners-budget', '{}', '{m1}', 1.5, '{}', false,
                    '2026-01-02 03:04:05.678', 'creator', '2026-02-03 04:05:06.789', 'updater')
            "#,
        )
        .execute(&mut **tx)
        .await;
        result.map(|_| ())
    }

    fn row(found: &Option<crate::RawJson>) -> Result<serde_json::Value, Box<dyn StdError>> {
        Ok(serde_json::from_str(
            found.as_ref().ok_or("row missing")?.get(),
        )?)
    }

    #[rstest]
    #[tokio::test]
    #[ignore = "requires DATABASE_URL pointing to a Prisma-migrated PostgreSQL database"]
    async fn reads_each_owner_asked_for(
        #[future(awt)] database_pool: Result<PgPool, Box<dyn StdError>>,
    ) -> Result<(), Box<dyn StdError>> {
        let pool = database_pool?;
        let mut tx = pool.begin().await?;
        insert_owners(&mut tx).await?;

        let found = find_key_owners(
            &mut *tx,
            Some("owners-user"),
            Some("owners-team"),
            Some("owners-user"),
            Some("owners-org-missing"),
            Some("owners-project"),
        )
        .await?;

        let user = row(&found.user_row)?;
        assert_eq!(user["user_id"], "owners-user");
        assert_eq!(user["models"], serde_json::json!([]));
        assert_eq!(user["organization_memberships"], serde_json::json!([]));
        let team = row(&found.team_row)?;
        assert_eq!(team["team_id"], "owners-team");
        assert_eq!(team["litellm_model_table"], serde_json::Value::Null);
        assert_eq!(team["object_permission"], serde_json::Value::Null);
        assert!(found.membership_row.is_none());
        assert!(found.organization_row.is_none());
        let project = row(&found.project_row)?;
        assert_eq!(project["models"], serde_json::json!(["m1"]));
        assert_eq!(
            project["litellm_budget_table"]["budget_id"],
            "owners-budget"
        );
        assert_eq!(
            project["litellm_budget_table"]["allowed_models"],
            serde_json::json!(["gpt-4o"])
        );
        Ok(())
    }

    #[rstest]
    #[tokio::test]
    #[ignore = "requires DATABASE_URL pointing to a Prisma-migrated PostgreSQL database"]
    async fn nothing_asked_for_is_all_none(
        #[future(awt)] database_pool: Result<PgPool, Box<dyn StdError>>,
    ) -> Result<(), Box<dyn StdError>> {
        let pool = database_pool?;
        let found = find_key_owners(&pool, None, None, None, None, None).await?;
        assert_eq!(
            serde_json::to_value(&found)?,
            serde_json::json!({
                "user_row": null,
                "team_row": null,
                "membership_row": null,
                "organization_row": null,
                "project_row": null,
            })
        );
        Ok(())
    }
}
