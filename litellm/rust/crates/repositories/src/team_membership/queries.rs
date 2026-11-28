//! `LiteLLM_TeamMembership` reads for auth, with the member's `LiteLLM_BudgetTable` row nested in by the SQL, and
//! the statements of a member add.

use serde::Serialize;
use sqlx::{PgExecutor, types::Json};

use litellm_db::Error;

use super::repository::{NewCappedMemberBudget, NewMemberBudget, RosterRole};
use crate::RawJson;
use crate::budget::Budget;
use crate::utc_timestamp::naive_utc;

/// A user's membership of a team, with the per-member budget when one is assigned, as the host's
/// `LiteLLM_TeamMembership` model reads it
#[macro_rules_attribute::apply(crate::db_row)]
#[derive(Serialize)]
pub struct TeamMembership {
    pub user_id: String,
    pub team_id: String,
    pub budget_id: Option<String>,
    pub spend: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rollover_credit: Option<f64>,
    pub total_spend: f64,
    pub litellm_budget_table: Option<Json<Budget>>,
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

/// [`insert_member_budget`] for a budget with a cap of its own.
pub(super) async fn insert_capped_member_budget(
    executor: impl PgExecutor<'_>,
    budget: &NewCappedMemberBudget,
    user_id: &str,
    team_id: &str,
) -> Result<(), Error> {
    sqlx::query_file!(
        "src/team_membership/queries/insert_capped_member_budget.sql",
        budget.budget_id,
        budget.max_budget,
        budget.allowed_models.as_deref(),
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

/// [`append_member_to_roster`] with the entry's `role`.
pub(super) async fn append_member_as_to_roster(
    executor: impl PgExecutor<'_>,
    team_id: &str,
    user_id: &str,
    user_email: Option<&str>,
    role: RosterRole,
) -> Result<serde_json::Value, Error> {
    let Json(roster) = sqlx::query_file_scalar!(
        "src/team_membership/queries/append_member_as_to_roster.sql",
        team_id,
        user_id,
        user_email,
        <&'static str>::from(role)
    )
    .fetch_one(executor)
    .await?;
    Ok(roster)
}

/// The roster of `team_id` as stored, or `None` when the team is gone.
pub(super) async fn find_roster(
    executor: impl PgExecutor<'_>,
    team_id: &str,
) -> Result<Option<RawJson>, Error> {
    Ok(
        sqlx::query_file_scalar!("src/team_membership/queries/find_roster.sql", team_id)
            .fetch_optional(executor)
            .await?,
    )
}

/// Whether the row of `user_id` lists `team_id` among its teams, or `None` when the user is gone.
pub(super) async fn user_holds_team(
    executor: impl PgExecutor<'_>,
    user_id: &str,
    team_id: &str,
) -> Result<Option<bool>, Error> {
    Ok(sqlx::query_file_scalar!(
        "src/team_membership/queries/user_holds_team.sql",
        user_id,
        team_id
    )
    .fetch_optional(executor)
    .await?)
}

/// Drops every roster entry of `team_id` naming `user_id` and returns the roster as stored.
pub(super) async fn remove_member_from_roster(
    executor: impl PgExecutor<'_>,
    team_id: &str,
    user_id: &str,
) -> Result<RawJson, Error> {
    Ok(sqlx::query_file_scalar!(
        "src/team_membership/queries/remove_member_from_roster.sql",
        team_id,
        user_id
    )
    .fetch_one(executor)
    .await?)
}

/// Takes `team_id` off the teams of `user_id`, when the row lists it.
pub(super) async fn remove_team_from_user(
    executor: impl PgExecutor<'_>,
    user_id: &str,
    team_id: &str,
) -> Result<(), Error> {
    sqlx::query_file!(
        "src/team_membership/queries/remove_team_from_user.sql",
        user_id,
        team_id
    )
    .execute(executor)
    .await?;
    Ok(())
}

/// Deletes the membership row of `user_id` in `team_id`, if there is one.
pub(super) async fn delete_team_membership(
    executor: impl PgExecutor<'_>,
    user_id: &str,
    team_id: &str,
) -> Result<(), Error> {
    sqlx::query_file!(
        "src/team_membership/queries/delete_team_membership.sql",
        user_id,
        team_id
    )
    .execute(executor)
    .await?;
    Ok(())
}

/// A key of the member's that a removal archived and deleted, once per JWT claim mapped to it, or once with no
/// claim.
#[macro_rules_attribute::apply(crate::db_row)]
pub(super) struct DeletedKeyClaim {
    pub token: String,
    pub jwt_issuer: Option<String>,
    pub jwt_claim_name: Option<String>,
    pub jwt_claim_value: Option<String>,
}

/// Deletes the keys `user_id` holds in `team_id` and archives the same rows to `LiteLLM_DeletedVerificationToken`,
/// stamped in UTC, in one statement, so a key created meanwhile is never deleted without its archive row. Their JWT
/// mappings go with them, and come back as they stood.
pub(super) async fn archive_and_delete_member_keys(
    executor: impl PgExecutor<'_>,
    user_id: &str,
    team_id: &str,
) -> Result<Vec<DeletedKeyClaim>, Error> {
    Ok(sqlx::query_file_as!(
        DeletedKeyClaim,
        "src/team_membership/queries/archive_and_delete_member_keys.sql",
        user_id,
        team_id
    )
    .fetch_all(executor)
    .await?)
}

#[cfg(test)]
mod tests {
    use std::error::Error as StdError;

    use super::find_team_membership;
    use crate::team_membership::repository::{
        AddedMember, JoinBudget, MemberAdd, MemberBudget, MemberRemove, NewCappedMemberBudget,
        NewMemberBudget, RefusedRoster, RosterRole, add_member_as_once, add_member_once,
        remove_member_once,
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

    /// One transaction for every step that, from the removal's key step on, creates a key of the member's in the team
    /// before each step, the way keys generated during the removal would land, since key creation takes no team lock.
    /// A removal that archived and deleted in separate steps would delete the key created between them unarchived
    struct KeyCreatedBeforeKeyStep<'t, 'c> {
        tx: &'t mut Transaction<'c, Postgres>,
        steps: usize,
    }

    impl crate::database::Connections for KeyCreatedBeforeKeyStep<'_, '_> {
        async fn with<T>(
            &mut self,
            _side: crate::Side,
            step: impl for<'e> AsyncFnOnce(
                litellm_db::Statements<&'e mut sqlx::PgConnection>,
            ) -> Result<T, litellm_db::Error>,
        ) -> Result<T, litellm_db::Error> {
            if self.steps >= 6 {
                #[expect(
                    clippy::disallowed_methods,
                    reason = "Fixture inserts are outside the compile-checked query set"
                )]
                let created = sqlx::query(
                    r#"INSERT INTO "LiteLLM_VerificationToken" (token, user_id, team_id)
                       VALUES ($1, 'member-add-user', 'member-add-team')"#,
                )
                .bind(format!("created-before-step-{}", self.steps));
                created.execute(&mut **self.tx).await?;
            }
            self.steps += 1;
            step(litellm_db::Statements::new(&mut **self.tx, false)).await
        }
    }

    #[rstest]
    #[tokio::test]
    #[ignore = "requires DATABASE_URL pointing to a Prisma-migrated PostgreSQL database"]
    async fn member_removal_archives_every_key_it_deletes(
        #[future(awt)] database_pool: Result<PgPool, Box<dyn StdError>>,
    ) -> Result<(), Box<dyn StdError>> {
        let pool = database_pool?;
        let mut tx = pool.begin().await?;
        insert_team_and_user(
            &mut tx,
            serde_json::json!([{"user_id": "member-add-user", "user_email": null, "role": "user"}]),
            None,
            &["member-add-team"],
        )
        .await?;

        let removed = remove_member_once(
            &mut KeyCreatedBeforeKeyStep {
                tx: &mut tx,
                steps: 0,
            },
            "member-add-user",
            "member-add-team",
            &|_| true,
        )
        .await?;
        #[expect(
            clippy::disallowed_methods,
            reason = "Assertion reads are outside the compile-checked query set"
        )]
        let archived: Vec<String> = sqlx::query_scalar(
            r#"SELECT token FROM "LiteLLM_DeletedVerificationToken" ORDER BY token"#,
        )
        .fetch_all(&mut *tx)
        .await?;

        let Ok(MemberRemove::Removed(removed)) = removed else {
            return Err(format!("the member was removed: {removed:?}").into());
        };
        assert_eq!(
            (removed.deleted_tokens, archived),
            (
                vec!["created-before-step-6".to_owned()],
                vec!["created-before-step-6".to_owned()]
            )
        );
        Ok(())
    }

    /// Python's own rows for the fixture `insert_captured_rows` stores, captured on 2026-11-27 by running
    /// `scim_v2._remove_user_from_team("leaver", "roster")` (which calls `team_member_delete`) and
    /// `get_user_object(..., user_id_upsert=True)` with `default_internal_user_params.teams` set to
    /// `[{"team_id": "dt-capped", "max_budget_in_team": 3.0, "user_role": "admin"}, {"team_id": "dt-plain"}]` against a
    /// migrated Postgres, then `row_to_json` of each row, less its generated id and times. Prisma reads the key's NULL
    /// array columns as `[]`, which is what Python archives.
    const PYTHON_ARCHIVED_KEY: &str = concat!(
        r#"{"access_group_ids":[],"agent_id":null,"aliases":{},"allowed_cache_controls":[],"#,
        r#""allowed_routes":[],"auto_rotate":false,"blocked":null,"budget_duration":null,"#,
        r#""budget_fallbacks":{},"budget_id":null,"budget_reset_at":null,"config":{},"created_by":null,"#,
        r#""deleted_by":null,"deleted_by_api_key":null,"expires":null,"key_alias":"kept alias","#,
        r#""key_id":null,"key_name":null,"key_rotation_at":null,"key_type":null,"last_active":null,"#,
        r#""last_rotation_at":null,"litellm_changed_by":null,"max_budget":null,"#,
        r#""max_parallel_requests":null,"metadata":{},"model_max_budget":{},"model_spend":{},"#,
        r#""models":[],"object_permission_id":null,"organization_id":null,"permissions":{},"#,
        r#""policies":[],"project_id":null,"rotation_count":0,"rotation_interval":null,"#,
        r#""router_settings":{},"rpm_limit":null,"settings_updated_at":null,"#,
        r#""soft_budget_cooldown":false,"spend":1.5,"team_id":"roster","token":"leaver-roster-key","#,
        r#""total_spend":0,"tpd_limit":null,"tpm_limit":null,"updated_by":null,"user_id":"leaver"}"#,
    );

    const PYTHON_CAPPED_MEMBER_BUDGET: &str = concat!(
        r#"{"allowed_models":["claude"],"budget_duration":null,"budget_limits":null,"#,
        r#""budget_reset_at":null,"budget_reset_count":0,"cap_policy":null,"#,
        r#""created_by":"default_user_id","max_budget":3,"max_parallel_requests":null,"#,
        r#""model_max_budget":null,"model_rpm_limit":null,"model_tpm_limit":null,"name":null,"#,
        r#""rollover_max_budget":null,"rpm_limit":null,"soft_budget":null,"temp_budget_expiry":null,"#,
        r#""temp_budget_increase":null,"tpd_limit":null,"tpm_limit":null,"updated_by":"default_user_id"}"#,
    );

    const PYTHON_CAPPED_ROSTER: &str =
        r#"[{"role": "admin", "user_id": "joiner", "user_email": "joiner@example.com"}]"#;

    const PYTHON_PLAIN_ROSTER: &str =
        r#"[{"role": "user", "user_id": "joiner", "user_email": "joiner@example.com"}]"#;

    /// The rows the capture above ran on: `roster` lists `other` and then `leaver`, whose row lists `roster` and
    /// `elsewhere` and who holds a key in each, the one in `roster` with every nullable array NULL and mapped to a
    /// JWT claim. `stale` lists `roster` with no roster entry, `joiner` is in no team, and `broken` lists `leaver`
    /// with no role, which `Member` refuses.
    async fn insert_captured_rows(tx: &mut Transaction<'_, Postgres>) -> Result<(), sqlx::Error> {
        let statements = [
            r#"INSERT INTO "LiteLLM_TeamTable" (team_id, members_with_roles, default_team_member_models, created_at, updated_at)
               VALUES ('roster', '[{"user_id": "other", "user_email": null, "role": "admin"},
                                   {"user_id": "leaver", "user_email": "leaver@example.com", "role": "user"}]',
                       ARRAY[]::text[], '2026-11-01 00:00:00', '2026-11-01 00:00:00'),
                      ('elsewhere', '[]', ARRAY[]::text[], '2026-11-01 00:00:00', '2026-11-01 00:00:00'),
                      ('dt-capped', '[]', ARRAY['claude'], '2026-11-01 00:00:00', '2026-11-01 00:00:00'),
                      ('dt-plain', '[]', ARRAY[]::text[], '2026-11-01 00:00:00', '2026-11-01 00:00:00'),
                      ('broken', '[{"user_id": "leaver"}]', ARRAY[]::text[], '2026-11-01 00:00:00', '2026-11-01 00:00:00')"#,
            r#"INSERT INTO "LiteLLM_UserTable" (user_id, user_email, teams, created_at, updated_at)
               VALUES ('leaver', 'leaver@example.com', ARRAY['roster', 'elsewhere', 'broken'], '2026-11-01 00:00:00', '2026-11-01 00:00:00'),
                      ('stale', NULL, ARRAY['roster'], '2026-11-01 00:00:00', '2026-11-01 00:00:00'),
                      ('joiner', 'joiner@example.com', ARRAY[]::text[], '2026-11-01 00:00:00', '2026-11-01 00:00:00')"#,
            r#"INSERT INTO "LiteLLM_TeamMembership" (user_id, team_id)
               VALUES ('leaver', 'roster'), ('leaver', 'elsewhere'), ('leaver', 'broken')"#,
            r#"INSERT INTO "LiteLLM_VerificationToken"
                   (token, user_id, team_id, key_alias, spend, allowed_cache_controls, allowed_routes, policies, access_group_ids)
               VALUES ('leaver-roster-key', 'leaver', 'roster', 'kept alias', 1.5, NULL, NULL, NULL, NULL),
                      ('leaver-elsewhere-key', 'leaver', 'elsewhere', NULL, 0, '{}', '{}', '{}', '{}'),
                      ('leaver-broken-key', 'leaver', 'broken', NULL, 0, '{}', '{}', '{}', '{}')"#,
            r#"INSERT INTO "LiteLLM_JWTKeyMapping" (id, jwt_claim_name, jwt_claim_value, token, jwt_issuer)
               VALUES ('mapping', 'sub', 'leaver-sub', 'leaver-roster-key', 'https://issuer.example.com')"#,
        ];
        for statement in statements {
            #[expect(
                clippy::disallowed_methods,
                reason = "Fixture inserts are outside the compile-checked query set"
            )]
            sqlx::query(statement).execute(&mut **tx).await?;
        }
        Ok(())
    }

    async fn text(
        tx: &mut Transaction<'_, Postgres>,
        sql: &'static str,
    ) -> Result<String, sqlx::Error> {
        #[expect(
            clippy::disallowed_methods,
            reason = "Assertion reads are outside the compile-checked query set"
        )]
        let read = sqlx::query_scalar(sql);
        read.fetch_one(&mut **tx).await
    }

    /// The JSON of the one row `sql` reads, less the columns generated per write
    async fn row_without(
        tx: &mut Transaction<'_, Postgres>,
        sql: &'static str,
        generated: &[&str],
    ) -> Result<serde_json::Value, Box<dyn StdError>> {
        let mut row: serde_json::Value = serde_json::from_str(&text(tx, sql).await?)?;
        let fields = row.as_object_mut().ok_or("the row is no object")?;
        generated.iter().for_each(|column| {
            fields.remove(*column);
        });
        Ok(row)
    }

    #[rstest]
    #[tokio::test]
    #[ignore = "requires DATABASE_URL pointing to a Prisma-migrated PostgreSQL database"]
    async fn member_removal_stores_the_rows_team_member_delete_stores(
        #[future(awt)] database_pool: Result<PgPool, Box<dyn StdError>>,
    ) -> Result<(), Box<dyn StdError>> {
        let pool = database_pool?;
        let mut tx = pool.begin().await?;
        insert_captured_rows(&mut tx).await?;
        leave_utc(&mut tx).await?;

        let removed =
            remove_member_once(&mut OneTransaction(&mut tx), "leaver", "roster", &|_| true).await?;

        let Ok(MemberRemove::Removed(removed)) = removed else {
            return Err(format!("the member was removed: {removed:?}").into());
        };
        assert_eq!(
            (
                removed.deleted_tokens,
                removed.mapped_claims,
                removed.roster_after.0.get().to_owned(),
            ),
            (
                vec!["leaver-roster-key".to_owned()],
                vec![crate::MappedClaim {
                    jwt_issuer: "https://issuer.example.com".to_owned(),
                    jwt_claim_name: "sub".to_owned(),
                    jwt_claim_value: "leaver-sub".to_owned(),
                }],
                r#"[{"role": "admin", "user_id": "other", "user_email": null}]"#.to_owned(),
            )
        );
        assert_eq!(
            (
                text(&mut tx, r#"SELECT members_with_roles::text FROM "LiteLLM_TeamTable" WHERE team_id = 'roster'"#).await?,
                text(&mut tx, r#"SELECT array_to_string(teams, ',') FROM "LiteLLM_UserTable" WHERE user_id = 'leaver'"#).await?,
                text(&mut tx, r#"SELECT string_agg(team_id, ',' ORDER BY team_id) FROM "LiteLLM_TeamMembership" WHERE user_id = 'leaver'"#).await?,
                text(&mut tx, r#"SELECT string_agg(token, ',' ORDER BY token) FROM "LiteLLM_VerificationToken""#).await?,
                text(&mut tx, r#"SELECT count(*)::text FROM "LiteLLM_JWTKeyMapping""#).await?,
                row_without(
                    &mut tx,
                    r#"SELECT row_to_json(d)::text FROM "LiteLLM_DeletedVerificationToken" AS d"#,
                    &["id", "created_at", "updated_at", "deleted_at"],
                )
                .await?,
            ),
            (
                r#"[{"role": "admin", "user_id": "other", "user_email": null}]"#.to_owned(),
                "elsewhere,broken".to_owned(),
                "broken,elsewhere".to_owned(),
                "leaver-broken-key,leaver-elsewhere-key".to_owned(),
                "0".to_owned(),
                serde_json::from_str::<serde_json::Value>(PYTHON_ARCHIVED_KEY)?,
            )
        );
        assert_eq!(
            text(
                &mut tx,
                r#"SELECT string_agg((abs(extract(epoch FROM stamped - (now() AT TIME ZONE 'UTC'))) < 1)::text, ',')
                   FROM (SELECT updated_at AS stamped FROM "LiteLLM_UserTable" WHERE user_id = 'leaver'
                         UNION ALL SELECT updated_at FROM "LiteLLM_TeamTable" WHERE team_id = 'roster'
                         UNION ALL SELECT deleted_at FROM "LiteLLM_DeletedVerificationToken") AS stamps"#,
            )
            .await?,
            "true,true,true"
        );
        Ok(())
    }

    #[rstest]
    #[tokio::test]
    #[ignore = "requires DATABASE_URL pointing to a Prisma-migrated PostgreSQL database"]
    async fn member_removal_clears_a_team_only_the_users_row_lists(
        #[future(awt)] database_pool: Result<PgPool, Box<dyn StdError>>,
    ) -> Result<(), Box<dyn StdError>> {
        let pool = database_pool?;
        let mut tx = pool.begin().await?;
        insert_captured_rows(&mut tx).await?;

        let removed =
            remove_member_once(&mut OneTransaction(&mut tx), "stale", "roster", &|_| true).await?;

        let Ok(MemberRemove::Removed(removed)) = removed else {
            return Err(format!("the stale team was cleared: {removed:?}").into());
        };
        assert_eq!(
            (
                removed.roster_before == removed.roster_after,
                text(&mut tx, r#"SELECT array_to_string(teams, ',') FROM "LiteLLM_UserTable" WHERE user_id = 'stale'"#).await?,
            ),
            (true, String::new())
        );
        Ok(())
    }

    #[rstest]
    #[case::a_user_the_team_does_not_list("joiner", "roster", MemberRemove::NotMember)]
    #[case::a_team_no_row_has("leaver", "gone", MemberRemove::TeamMissing)]
    #[tokio::test]
    #[ignore = "requires DATABASE_URL pointing to a Prisma-migrated PostgreSQL database"]
    async fn member_removal_that_cannot_remove_writes_nothing(
        #[case] user_id: &str,
        #[case] team_id: &str,
        #[case] expected: MemberRemove,
        #[future(awt)] database_pool: Result<PgPool, Box<dyn StdError>>,
    ) -> Result<(), Box<dyn StdError>> {
        let pool = database_pool?;
        let mut tx = pool.begin().await?;
        insert_captured_rows(&mut tx).await?;

        let removed =
            remove_member_once(&mut OneTransaction(&mut tx), user_id, team_id, &|_| true).await?;

        assert_eq!(
            (
                removed,
                text(
                    &mut tx,
                    r#"SELECT count(*)::text FROM "LiteLLM_VerificationToken""#
                )
                .await?,
            ),
            (Ok(expected), "3".to_owned())
        );
        Ok(())
    }

    /// The host joins a user `user_id_upsert` created to each default team as Python does: an admin with a capped
    /// budget of their own taking the team's default member models, and a plain member with no budget
    #[rstest]
    #[tokio::test]
    #[ignore = "requires DATABASE_URL pointing to a Prisma-migrated PostgreSQL database"]
    async fn default_team_join_stores_the_rows_python_stores(
        #[future(awt)] database_pool: Result<PgPool, Box<dyn StdError>>,
    ) -> Result<(), Box<dyn StdError>> {
        let pool = database_pool?;
        let mut tx = pool.begin().await?;
        insert_captured_rows(&mut tx).await?;
        let capped = NewCappedMemberBudget {
            budget_id: "joiner-capped".to_owned(),
            max_budget: 3.0,
            allowed_models: Some(vec!["claude".to_owned()]),
            created_by: "default_user_id".to_owned(),
            updated_by: "default_user_id".to_owned(),
        };

        let as_admin = add_member_as_once(
            &mut OneTransaction(&mut tx),
            "joiner",
            "dt-capped",
            RosterRole::Admin,
            JoinBudget::Capped(&capped),
            &|_| true,
        )
        .await?;
        let as_user = add_member_as_once(
            &mut OneTransaction(&mut tx),
            "joiner",
            "dt-plain",
            RosterRole::User,
            JoinBudget::Member(MemberBudget::None),
            &|_| true,
        )
        .await?;

        assert!(
            matches!(
                (&as_admin, &as_user),
                (Ok(MemberAdd::Added(_)), Ok(MemberAdd::Added(_)))
            ),
            "{as_admin:?} {as_user:?}"
        );
        assert_eq!(
            (
                text(&mut tx, r#"SELECT members_with_roles::text FROM "LiteLLM_TeamTable" WHERE team_id = 'dt-capped'"#).await?,
                text(&mut tx, r#"SELECT members_with_roles::text FROM "LiteLLM_TeamTable" WHERE team_id = 'dt-plain'"#).await?,
                text(&mut tx, r#"SELECT array_to_string(teams, ',') FROM "LiteLLM_UserTable" WHERE user_id = 'joiner'"#).await?,
                text(
                    &mut tx,
                    r#"SELECT COALESCE(budget_id, 'none') FROM "LiteLLM_TeamMembership"
                       WHERE user_id = 'joiner' AND team_id = 'dt-plain'"#,
                )
                .await?,
                row_without(
                    &mut tx,
                    r#"SELECT row_to_json(b)::text FROM "LiteLLM_BudgetTable" AS b
                       JOIN "LiteLLM_TeamMembership" AS m ON m.budget_id = b.budget_id
                       WHERE m.user_id = 'joiner' AND m.team_id = 'dt-capped'"#,
                    &["budget_id", "created_at", "updated_at"],
                )
                .await?,
            ),
            (
                PYTHON_CAPPED_ROSTER.to_owned(),
                PYTHON_PLAIN_ROSTER.to_owned(),
                "dt-capped,dt-plain".to_owned(),
                "none".to_owned(),
                serde_json::from_str::<serde_json::Value>(PYTHON_CAPPED_MEMBER_BUDGET)?,
            )
        );
        Ok(())
    }

    #[rstest]
    #[case::an_add("joiner", true)]
    #[case::a_removal("leaver", false)]
    #[tokio::test]
    #[ignore = "requires DATABASE_URL pointing to a Prisma-migrated PostgreSQL database"]
    async fn a_roster_the_caller_refuses_is_handed_back_with_nothing_written(
        #[case] user_id: &str,
        #[case] adding: bool,
        #[future(awt)] database_pool: Result<PgPool, Box<dyn StdError>>,
    ) -> Result<(), Box<dyn StdError>> {
        let pool = database_pool?;
        let mut tx = pool.begin().await?;
        insert_captured_rows(&mut tx).await?;

        let refused = match adding {
            true => add_member_as_once(
                &mut OneTransaction(&mut tx),
                user_id,
                "broken",
                RosterRole::User,
                JoinBudget::Member(MemberBudget::None),
                &|_| false,
            )
            .await?
            .err(),
            false => {
                remove_member_once(&mut OneTransaction(&mut tx), user_id, "broken", &|_| false)
                    .await?
                    .err()
            }
        };

        assert_eq!(
            (
                refused.map(|RefusedRoster(roster)| roster.0.get().to_owned()),
                text(&mut tx, r#"SELECT members_with_roles::text FROM "LiteLLM_TeamTable" WHERE team_id = 'broken'"#).await?,
                text(&mut tx, r#"SELECT string_agg(token, ',' ORDER BY token) FROM "LiteLLM_VerificationToken" WHERE team_id = 'broken'"#).await?,
                text(&mut tx, r#"SELECT count(*)::text FROM "LiteLLM_TeamMembership" WHERE team_id = 'broken'"#).await?,
            ),
            (
                Some(r#"[{"user_id": "leaver"}]"#.to_owned()),
                r#"[{"user_id": "leaver"}]"#.to_owned(),
                "leaver-broken-key".to_owned(),
                "1".to_owned(),
            )
        );
        Ok(())
    }
}
