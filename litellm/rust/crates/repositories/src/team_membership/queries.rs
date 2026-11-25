//! `LiteLLM_TeamMembership` reads for auth, with the member's `LiteLLM_BudgetTable` row nested in by the SQL, and
//! the statements of a member add.

use serde::Serialize;
use sqlx::{PgExecutor, types::Json};

use litellm_db::Error;

use super::repository::NewMemberBudget;
use crate::budget::Budget;
use crate::utc_timestamp::naive_utc;

/// A user's membership of a team, with the per-member budget when one is assigned, as the host's
/// `LiteLLM_TeamMembership` model reads it
#[macro_rules_attribute::apply(crate::db_row)]
#[derive(Serialize)]
pub struct TeamMembership {
    user_id: String,
    team_id: String,
    budget_id: Option<String>,
    spend: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    rollover_credit: Option<f64>,
    total_spend: f64,
    litellm_budget_table: Option<Json<Budget>>,
}

impl TeamMembership {
    /// The `spend` column, what a budget counter for this row is seeded and floored from.
    pub fn spend(&self) -> f64 {
        self.spend
    }
}

/// `user_id`'s membership of `team_id`, or `None` when the user is not a member.
pub(super) async fn find_team_membership(
    executor: impl PgExecutor<'_>,
    user_id: &str,
    team_id: &str,
) -> Result<Option<TeamMembership>, Error> {
    Ok(sqlx::query_file_as!(
        TeamMembership,
        "src/team_membership/queries/find_team_membership.sql",
        user_id,
        team_id,
    )
    .fetch_optional(executor)
    .await?)
}

/// Takes the advisory lock the management endpoints hold while they rewrite the roster of `team_id`, until the
/// transaction ends.
pub(super) async fn lock_team(executor: impl PgExecutor<'_>, team_id: &str) -> Result<(), Error> {
    sqlx::query_file_scalar!("src/team_membership/queries/lock_team.sql", team_id)
        .fetch_one(executor)
        .await?;
    Ok(())
}

/// Whether the roster of `team_id` lists `user_id` or that user's email, or `None` when the team is gone.
pub(super) async fn roster_lists_user(
    executor: impl PgExecutor<'_>,
    team_id: &str,
    user_id: &str,
) -> Result<Option<bool>, Error> {
    Ok(sqlx::query_file_scalar!(
        "src/team_membership/queries/roster_lists_user.sql",
        team_id,
        user_id
    )
    .fetch_optional(executor)
    .await?)
}

/// Locks the row of `user_id` and reads its email, or `None` when the user is gone.
pub(super) async fn lock_user(
    executor: impl PgExecutor<'_>,
    user_id: &str,
) -> Result<Option<Option<String>>, Error> {
    Ok(
        sqlx::query_file_scalar!("src/team_membership/queries/lock_user.sql", user_id)
            .fetch_optional(executor)
            .await?,
    )
}

/// Puts `team_id` on the user's `teams` unless it is there already.
pub(super) async fn add_team_to_user(
    executor: impl PgExecutor<'_>,
    user_id: &str,
    team_id: &str,
) -> Result<(), Error> {
    sqlx::query_file!(
        "src/team_membership/queries/add_team_to_user.sql",
        user_id,
        team_id
    )
    .execute(executor)
    .await?;
    Ok(())
}

/// Inserts `budget` for the membership about to be written, unless `user_id` already has one in `team_id`, which
/// keeps its own budget.
pub(super) async fn insert_member_budget(
    executor: impl PgExecutor<'_>,
    budget: &NewMemberBudget,
    user_id: &str,
    team_id: &str,
) -> Result<(), Error> {
    sqlx::query_file!(
        "src/team_membership/queries/insert_member_budget.sql",
        budget.budget_id,
        &budget.allowed_models,
        budget.budget_duration,
        budget.budget_reset_at.and_then(naive_utc),
        budget.created_by,
        budget.updated_by,
        user_id,
        team_id
    )
    .execute(executor)
    .await?;
    Ok(())
}

/// The membership of `user_id` in `team_id`, inserted with `budget_id` unless the user already has one, which is
/// left as it is.
pub(super) async fn insert_team_membership(
    executor: impl PgExecutor<'_>,
    user_id: &str,
    team_id: &str,
    budget_id: Option<&str>,
) -> Result<TeamMembership, Error> {
    Ok(sqlx::query_file_as!(
        TeamMembership,
        "src/team_membership/queries/insert_team_membership.sql",
        user_id,
        team_id,
        budget_id
    )
    .fetch_one(executor)
    .await?)
}

/// Appends the user to the roster of `team_id` as a `user`, in the shape of the host's `Member` model. The roster
/// as stored, the new entry last.
pub(super) async fn append_member_to_roster(
    executor: impl PgExecutor<'_>,
    team_id: &str,
    user_id: &str,
    user_email: Option<&str>,
) -> Result<serde_json::Value, Error> {
    let Json(roster) = sqlx::query_file_scalar!(
        "src/team_membership/queries/append_member_to_roster.sql",
        team_id,
        user_id,
        user_email
    )
    .fetch_one(executor)
    .await?;
    Ok(roster)
}

#[cfg(test)]
mod tests {
    use std::error::Error as StdError;

    use super::find_team_membership;
    use crate::team_membership::repository::{
        AddedMember, MemberAdd, MemberBudget, NewMemberBudget, add_member_once,
    };
    use crate::test_support::{OneTransaction, database_pool, insert_budget, leave_utc};
    use rstest::rstest;
    use sqlx::{PgPool, Postgres, Transaction};

    #[rstest]
    #[case::with_member_budget(Some("member-budget"))]
    #[case::without_member_budget(None)]
    #[tokio::test]
    #[ignore = "requires DATABASE_URL pointing to a Prisma-migrated PostgreSQL database"]
    async fn reads_membership_with_member_budget(
        #[case] budget_id: Option<&str>,
        #[future(awt)] database_pool: Result<PgPool, Box<dyn StdError>>,
    ) -> Result<(), Box<dyn StdError>> {
        let pool = database_pool?;
        let mut tx = pool.begin().await?;
        insert_budget(&mut tx, "member-budget").await?;
        #[expect(
            clippy::disallowed_methods,
            reason = "Fixture writes are outside the compile-checked query set"
        )]
        let _inserted = sqlx::query(
            r#"INSERT INTO "LiteLLM_TeamMembership" (user_id, team_id, spend, total_spend, budget_id)
               VALUES ('member', 'member-team', 1.5, 2.5, $1)"#,
        )
        .bind(budget_id)
        .execute(&mut *tx)
        .await?;

        let membership = find_team_membership(&mut *tx, "member", "member-team")
            .await?
            .ok_or("membership missing")?;

        assert_eq!(membership.budget_id.as_deref(), budget_id);
        assert_eq!(membership.spend, 1.5);
        assert_eq!(membership.total_spend, 2.5);
        assert_eq!(
            membership
                .litellm_budget_table
                .map(|budget| (budget.rpm_limit, budget.allowed_models.clone())),
            budget_id.map(|_| (Some(10), Some(vec!["gpt-4o".to_owned()])))
        );
        Ok(())
    }

    #[rstest]
    #[tokio::test]
    #[ignore = "requires DATABASE_URL pointing to a Prisma-migrated PostgreSQL database"]
    async fn missing_membership_is_none(
        #[future(awt)] database_pool: Result<PgPool, Box<dyn StdError>>,
    ) -> Result<(), Box<dyn StdError>> {
        let pool = database_pool?;
        assert!(
            find_team_membership(&pool, "no-such-user", "no-such-team")
                .await?
                .is_none()
        );
        Ok(())
    }

    /// Team `member-add-team` with `roster`, user `member-add-user` with `user_email` and `teams`, the budget
    /// `member-budget`, and a bystander team and user no member add names
    async fn insert_team_and_user(
        tx: &mut Transaction<'_, Postgres>,
        roster: serde_json::Value,
        user_email: Option<&str>,
        teams: &[&str],
    ) -> Result<(), sqlx::Error> {
        insert_budget(tx, "member-budget").await?;
        #[expect(
            clippy::disallowed_methods,
            reason = "Fixture inserts are outside the compile-checked query set"
        )]
        sqlx::query(
            r#"
            INSERT INTO "LiteLLM_TeamTable" (team_id, members_with_roles, created_at, updated_at)
            VALUES ('member-add-team', $1, '2026-11-01 00:00:00', '2026-11-01 00:00:00'),
                   ('member-add-bystander-team', '[]', '2026-11-01 00:00:00', '2026-11-01 00:00:00')
            "#,
        )
        .bind(roster)
        .execute(&mut **tx)
        .await?;
        #[expect(
            clippy::disallowed_methods,
            reason = "Fixture inserts are outside the compile-checked query set"
        )]
        let result = sqlx::query(
            r#"
            INSERT INTO "LiteLLM_UserTable" (user_id, user_email, teams, created_at, updated_at)
            VALUES ('member-add-user', $1, $2, '2026-11-01 00:00:00', '2026-11-01 00:00:00'),
                   ('member-add-bystander', 'bystander@example.com', '{}', '2026-11-01 00:00:00',
                    '2026-11-01 00:00:00')
            "#,
        )
        .bind(user_email)
        .bind(teams)
        .execute(&mut **tx)
        .await;
        result.map(|_| ())
    }

    /// What a member add can write: the user's `teams`, the team's roster, the membership's budget when the row
    /// exists, and whether the user row and the team row were stamped at UTC now
    #[derive(Debug, PartialEq, sqlx::FromRow)]
    struct Stored {
        teams: Option<Vec<String>>,
        roster: Option<sqlx::types::Json<serde_json::Value>>,
        membership_budgets: Vec<Option<String>>,
        own_budgets: i64,
        user_stamped: Option<bool>,
        team_stamped: Option<bool>,
        bystander_teams: Option<Vec<String>>,
        bystander_roster: Option<sqlx::types::Json<serde_json::Value>>,
    }

    async fn stored(tx: &mut Transaction<'_, Postgres>) -> Result<Stored, sqlx::Error> {
        #[expect(
            clippy::disallowed_methods,
            reason = "Assertion reads are outside the compile-checked query set"
        )]
        sqlx::query_as(
            r#"
            SELECT (SELECT teams FROM "LiteLLM_UserTable" WHERE user_id = 'member-add-user') AS teams,
                   (SELECT members_with_roles FROM "LiteLLM_TeamTable" WHERE team_id = 'member-add-team') AS roster,
                   ARRAY(SELECT budget_id FROM "LiteLLM_TeamMembership"
                         WHERE user_id = 'member-add-user' AND team_id = 'member-add-team') AS membership_budgets,
                   (SELECT count(*) FROM "LiteLLM_BudgetTable" WHERE budget_id = 'member-own-budget') AS own_budgets,
                   (SELECT abs(extract(epoch FROM updated_at - (now() AT TIME ZONE 'UTC'))) < 1
                    FROM "LiteLLM_UserTable" WHERE user_id = 'member-add-user') AS user_stamped,
                   (SELECT abs(extract(epoch FROM updated_at - (now() AT TIME ZONE 'UTC'))) < 1
                    FROM "LiteLLM_TeamTable" WHERE team_id = 'member-add-team') AS team_stamped,
                   (SELECT teams FROM "LiteLLM_UserTable" WHERE user_id = 'member-add-bystander') AS bystander_teams,
                   (SELECT members_with_roles FROM "LiteLLM_TeamTable"
                    WHERE team_id = 'member-add-bystander-team') AS bystander_roster
            "#,
        )
        .fetch_one(&mut **tx)
        .await
    }

    fn roster(entries: serde_json::Value) -> Option<sqlx::types::Json<serde_json::Value>> {
        Some(sqlx::types::Json(entries))
    }

    #[rstest]
    #[case::beside_another_member(
        serde_json::json!([{"user_id": "other", "user_email": "other@example.com", "role": "admin"}]),
        Some("member@example.com"),
        serde_json::json!([
            {"user_id": "other", "user_email": "other@example.com", "role": "admin"},
            {"user_id": "member-add-user", "user_email": "member@example.com", "role": "user"},
        ])
    )]
    #[case::onto_the_legacy_empty_object_roster(
        serde_json::json!({}),
        None,
        serde_json::json!([{"user_id": "member-add-user", "user_email": null, "role": "user"}])
    )]
    #[tokio::test]
    #[ignore = "requires DATABASE_URL pointing to a Prisma-migrated PostgreSQL database"]
    async fn member_add_puts_the_team_on_the_user_a_membership_row_and_the_user_on_the_roster_stamped_in_utc(
        #[case] roster_before: serde_json::Value,
        #[case] user_email: Option<&str>,
        #[case] roster_after: serde_json::Value,
        #[future(awt)] database_pool: Result<PgPool, Box<dyn StdError>>,
    ) -> Result<(), Box<dyn StdError>> {
        let pool = database_pool?;
        let mut tx = pool.begin().await?;
        insert_team_and_user(&mut tx, roster_before, user_email, &["earlier-team"]).await?;
        leave_utc(&mut tx).await?;

        let added = add_member_once(
            &mut OneTransaction(&mut tx),
            "member-add-user",
            "member-add-team",
            MemberBudget::None,
        )
        .await?;

        let membership = find_team_membership(&mut *tx, "member-add-user", "member-add-team")
            .await?
            .ok_or("membership row missing")?;
        assert_eq!(
            added,
            MemberAdd::Added(Box::new(AddedMember {
                membership,
                roster: roster_after.clone(),
            }))
        );
        assert_eq!(
            stored(&mut tx).await?,
            Stored {
                teams: Some(vec![
                    "earlier-team".to_owned(),
                    "member-add-team".to_owned()
                ]),
                roster: roster(roster_after),
                membership_budgets: vec![None],
                own_budgets: 0,
                user_stamped: Some(true),
                team_stamped: Some(true),
                bystander_teams: Some(Vec::new()),
                bystander_roster: roster(serde_json::json!([])),
            }
        );
        Ok(())
    }

    #[rstest]
    #[tokio::test]
    #[ignore = "requires DATABASE_URL pointing to a Prisma-migrated PostgreSQL database"]
    async fn member_add_links_the_membership_to_the_budget_it_is_given(
        #[future(awt)] database_pool: Result<PgPool, Box<dyn StdError>>,
    ) -> Result<(), Box<dyn StdError>> {
        let pool = database_pool?;
        let mut tx = pool.begin().await?;
        insert_team_and_user(&mut tx, serde_json::json!([]), None, &[]).await?;

        let added = add_member_once(
            &mut OneTransaction(&mut tx),
            "member-add-user",
            "member-add-team",
            MemberBudget::Existing("member-budget"),
        )
        .await?;

        let MemberAdd::Added(stored_member) = added else {
            return Err(format!("expected an added member, got {added:?}").into());
        };
        let membership = stored_member.membership;
        let stored = serde_json::to_value(&membership)?;
        assert_eq!(
            (
                &stored["budget_id"],
                &stored["litellm_budget_table"]["max_budget"]
            ),
            (
                &serde_json::json!("member-budget"),
                &serde_json::json!(10.5)
            )
        );
        assert_eq!(
            Some(membership),
            find_team_membership(&mut *tx, "member-add-user", "member-add-team").await?
        );
        Ok(())
    }

    fn own_budget() -> Result<NewMemberBudget, jiff::Error> {
        Ok(NewMemberBudget {
            budget_id: "member-own-budget".to_owned(),
            allowed_models: vec!["gpt-4o".to_owned()],
            budget_duration: Some("30d".to_owned()),
            budget_reset_at: Some("2026-12-01T00:00:00.250Z".parse()?),
            created_by: "creator".to_owned(),
            updated_by: "updater".to_owned(),
        })
    }

    #[rstest]
    #[tokio::test]
    #[ignore = "requires DATABASE_URL pointing to a Prisma-migrated PostgreSQL database"]
    async fn member_add_inserts_a_budget_of_the_members_own_and_links_the_membership_to_it(
        #[future(awt)] database_pool: Result<PgPool, Box<dyn StdError>>,
    ) -> Result<(), Box<dyn StdError>> {
        let pool = database_pool?;
        let mut tx = pool.begin().await?;
        insert_team_and_user(&mut tx, serde_json::json!([]), None, &[]).await?;
        leave_utc(&mut tx).await?;

        let added = add_member_once(
            &mut OneTransaction(&mut tx),
            "member-add-user",
            "member-add-team",
            MemberBudget::New(&own_budget()?),
        )
        .await?;

        let MemberAdd::Added(stored_member) = added else {
            return Err(format!("expected an added member, got {added:?}").into());
        };
        #[expect(
            clippy::disallowed_methods,
            reason = "Assertion reads are outside the compile-checked query set"
        )]
        let budget: sqlx::types::Json<serde_json::Value> = sqlx::query_scalar(
            r#"
            SELECT jsonb_build_object(
                'allowed_models', allowed_models,
                'budget_duration', budget_duration,
                'budget_reset_at', budget_reset_at,
                'created_by', created_by,
                'updated_by', updated_by,
                'max_budget', max_budget,
                'budget_reset_count', budget_reset_count,
                'stamped', abs(extract(epoch FROM created_at - (now() AT TIME ZONE 'UTC'))) < 1
                    AND abs(extract(epoch FROM updated_at - (now() AT TIME ZONE 'UTC'))) < 1
            )
            FROM "LiteLLM_BudgetTable"
            WHERE budget_id = 'member-own-budget'
            "#,
        )
        .fetch_one(&mut *tx)
        .await?;
        assert_eq!(
            budget.0,
            serde_json::json!({
                "allowed_models": ["gpt-4o"],
                "budget_duration": "30d",
                "budget_reset_at": "2026-12-01T00:00:00.25",
                "created_by": "creator",
                "updated_by": "updater",
                "max_budget": null,
                "budget_reset_count": 0,
                "stamped": true,
            })
        );
        let membership = serde_json::to_value(&stored_member.membership)?;
        assert_eq!(
            (
                &membership["budget_id"],
                &membership["litellm_budget_table"]["allowed_models"]
            ),
            (
                &serde_json::json!("member-own-budget"),
                &serde_json::json!(["gpt-4o"])
            )
        );
        Ok(())
    }

    #[rstest]
    #[tokio::test]
    #[ignore = "requires DATABASE_URL pointing to a Prisma-migrated PostgreSQL database"]
    async fn member_add_leaves_a_team_already_on_the_user_and_a_stored_membership_with_its_budget_as_they_are(
        #[future(awt)] database_pool: Result<PgPool, Box<dyn StdError>>,
    ) -> Result<(), Box<dyn StdError>> {
        let pool = database_pool?;
        let mut tx = pool.begin().await?;
        insert_team_and_user(&mut tx, serde_json::json!([]), None, &["member-add-team"]).await?;
        #[expect(
            clippy::disallowed_methods,
            reason = "Fixture inserts are outside the compile-checked query set"
        )]
        sqlx::query(
            r#"
            INSERT INTO "LiteLLM_TeamMembership" (user_id, team_id, budget_id, spend)
            VALUES ('member-add-user', 'member-add-team', 'member-budget', 4.5)
            "#,
        )
        .execute(&mut *tx)
        .await?;

        let added = add_member_once(
            &mut OneTransaction(&mut tx),
            "member-add-user",
            "member-add-team",
            MemberBudget::New(&own_budget()?),
        )
        .await?;

        let MemberAdd::Added(stored_member) = added else {
            return Err(format!("expected an added member, got {added:?}").into());
        };
        let after = stored(&mut tx).await?;
        assert_eq!(
            (
                serde_json::to_value(&stored_member.membership)?["spend"].clone(),
                after.teams,
                after.membership_budgets,
                after.own_budgets
            ),
            (
                serde_json::json!(4.5),
                Some(vec!["member-add-team".to_owned()]),
                vec![Some("member-budget".to_owned())],
                0
            )
        );
        Ok(())
    }

    #[rstest]
    #[case::the_roster_lists_the_user_id(
        serde_json::json!([{"user_id": "member-add-user", "user_email": null, "role": "admin"}]),
        "member-add-user",
        "member-add-team",
        MemberAdd::AlreadyMember
    )]
    #[case::the_roster_lists_the_users_email_under_another_id(
        serde_json::json!([{"user_id": "other", "user_email": "member@example.com", "role": "user"}]),
        "member-add-user",
        "member-add-team",
        MemberAdd::AlreadyMember
    )]
    #[case::the_team_is_gone(serde_json::json!([]), "member-add-user", "member-add-gone", MemberAdd::TeamMissing)]
    #[case::the_user_is_gone(serde_json::json!([]), "member-add-gone", "member-add-team", MemberAdd::UserMissing)]
    #[tokio::test]
    #[ignore = "requires DATABASE_URL pointing to a Prisma-migrated PostgreSQL database"]
    async fn member_add_that_cannot_add_writes_nothing(
        #[case] roster_before: serde_json::Value,
        #[case] user_id: &str,
        #[case] team_id: &str,
        #[case] expected: MemberAdd,
        #[future(awt)] database_pool: Result<PgPool, Box<dyn StdError>>,
    ) -> Result<(), Box<dyn StdError>> {
        let pool = database_pool?;
        let mut tx = pool.begin().await?;
        insert_team_and_user(
            &mut tx,
            roster_before.clone(),
            Some("member@example.com"),
            &[],
        )
        .await?;

        let outcome = add_member_once(
            &mut OneTransaction(&mut tx),
            user_id,
            team_id,
            MemberBudget::New(&own_budget()?),
        )
        .await?;

        assert_eq!(
            (outcome, stored(&mut tx).await?),
            (
                expected,
                Stored {
                    teams: Some(Vec::new()),
                    roster: roster(roster_before),
                    membership_budgets: Vec::new(),
                    own_budgets: 0,
                    user_stamped: Some(false),
                    team_stamped: Some(false),
                    bystander_teams: Some(Vec::new()),
                    bystander_roster: roster(serde_json::json!([])),
                }
            )
        );
        Ok(())
    }

    #[rstest]
    #[tokio::test]
    #[ignore = "requires DATABASE_URL pointing to a Prisma-migrated PostgreSQL database"]
    async fn member_add_holds_the_advisory_lock_the_management_endpoints_take_on_the_team(
        #[future(awt)] database_pool: Result<PgPool, Box<dyn StdError>>,
    ) -> Result<(), Box<dyn StdError>> {
        let pool = database_pool?;
        let mut tx = pool.begin().await?;
        insert_team_and_user(&mut tx, serde_json::json!([]), None, &[]).await?;

        add_member_once(
            &mut OneTransaction(&mut tx),
            "member-add-user",
            "member-add-team",
            MemberBudget::None,
        )
        .await?;

        #[expect(
            clippy::disallowed_methods,
            reason = "Assertion reads are outside the compile-checked query set"
        )]
        let held: bool = sqlx::query_scalar(
            r#"
            SELECT EXISTS (
                SELECT 1
                FROM pg_locks
                WHERE locktype = 'advisory'
                  AND pid = pg_backend_pid()
                  AND granted
                  AND objsubid = 1
                  AND ((classid::bigint << 32) | objid::bigint) = hashtext('member-add-team')::bigint
            )
            "#,
        )
        .fetch_one(&mut *tx)
        .await?;
        assert!(held);
        Ok(())
    }
}
