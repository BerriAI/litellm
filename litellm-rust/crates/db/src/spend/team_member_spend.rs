use sqlx::{Postgres, Transaction};

use super::lock_team;
use crate::Error;

#[macro_rules_attribute::apply(crate::db_input)]
pub struct TeamMemberSpend {
    pub team_id: String,
    pub user_id: String,
    pub cost: f64,
}

pub async fn add_team_member_spend(
    transaction: &mut Transaction<'_, Postgres>,
    members: &[TeamMemberSpend],
) -> Result<(), Error> {
    if members.is_empty() {
        return Ok(());
    }
    let mut members = members.to_vec();
    members.sort_by(|left, right| {
        (&left.team_id, &left.user_id).cmp(&(&right.team_id, &right.user_id))
    });
    let mut teams: Vec<&str> = members
        .iter()
        .map(|member| member.team_id.as_str())
        .collect();
    teams.dedup();
    for team in teams {
        lock_team(transaction, team).await?;
    }
    let rows = serde_json::Value::Array(
        members
            .iter()
            .map(|member| {
                serde_json::json!({
                    "user_id": member.user_id,
                    "team_id": member.team_id,
                    "cost": member.cost,
                })
            })
            .collect(),
    );
    sqlx::query_file!("queries/spend/upsert_team_member_spend.sql", rows)
        .execute(&mut **transaction)
        .await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::error::Error as StdError;

    use super::super::begin;
    use super::{TeamMemberSpend, add_team_member_spend};
    use crate::test_support::database_pool;
    use rstest::rstest;
    use sqlx::PgPool;

    fn members(cost: f64) -> [TeamMemberSpend; 3] {
        [
            TeamMemberSpend {
                team_id: "member-spend-z".into(),
                user_id: "off-roster-existing".into(),
                cost: 2.0 * cost,
            },
            TeamMemberSpend {
                team_id: "member-spend-a".into(),
                user_id: "on-roster-missing".into(),
                cost,
            },
            TeamMemberSpend {
                team_id: "member-spend-a".into(),
                user_id: "off-roster-missing".into(),
                cost: 50.0 * cost,
            },
        ]
    }

    #[rstest]
    #[case::zero_then_quarter(0.25, 0.5, 3.0, 8.0)]
    #[case::zero_then_eighth(0.125, 0.25, 2.5, 7.5)]
    #[tokio::test]
    #[ignore = "requires DATABASE_URL pointing to a Prisma-migrated PostgreSQL database"]
    async fn membership_spend_preserves_roster_guard_and_fractional_costs(
        #[case] cost: f64,
        #[case] expected_created: f64,
        #[case] expected_spend: f64,
        #[case] expected_total: f64,
        #[future(awt)] database_pool: Result<PgPool, Box<dyn StdError>>,
    ) -> Result<(), Box<dyn StdError>> {
        let pool = database_pool?;
        let mut tx = begin(&pool).await?;
        #[expect(
            clippy::disallowed_methods,
            reason = "Fixture writes are outside the compile-checked query set"
        )]
        let _teams = sqlx::query(
            r#"INSERT INTO "LiteLLM_TeamTable" (team_id, members_with_roles)
               VALUES ('member-spend-a', '[{"user_id":"on-roster-missing","role":"user"}]'),
                      ('member-spend-z', '[]')"#,
        )
        .execute(&mut *tx)
        .await?;
        #[expect(
            clippy::disallowed_methods,
            reason = "Fixture writes are outside the compile-checked query set"
        )]
        let _members = sqlx::query(
            r#"INSERT INTO "LiteLLM_TeamMembership" (team_id, user_id, spend, total_spend)
               VALUES ('member-spend-z', 'off-roster-existing', 2, 7),
                      ('member-spend-a', 'untouched', 100, 200)"#,
        )
        .execute(&mut *tx)
        .await?;
        add_team_member_spend(&mut tx, &[]).await?;
        add_team_member_spend(&mut tx, &members(0.0)).await?;
        add_team_member_spend(&mut tx, &members(cost)).await?;
        add_team_member_spend(&mut tx, &members(cost)).await?;
        #[expect(
            clippy::disallowed_methods,
            reason = "Test assertions are outside the compile-checked query set"
        )]
        let rows: Vec<(String, String, f64, f64)> = sqlx::query_as(
            r#"SELECT team_id, user_id, spend, total_spend FROM "LiteLLM_TeamMembership"
               WHERE team_id IN ('member-spend-a', 'member-spend-z') ORDER BY team_id, user_id"#,
        )
        .fetch_all(&mut *tx)
        .await?;
        assert_eq!(
            rows,
            vec![
                (
                    "member-spend-a".into(),
                    "on-roster-missing".into(),
                    expected_created,
                    expected_created
                ),
                ("member-spend-a".into(), "untouched".into(), 100.0, 200.0),
                (
                    "member-spend-z".into(),
                    "off-roster-existing".into(),
                    expected_spend,
                    expected_total
                ),
            ]
        );
        tx.rollback().await?;
        #[expect(
            clippy::disallowed_methods,
            reason = "Test assertions are outside the compile-checked query set"
        )]
        let remaining: i64 = sqlx::query_scalar(
            r#"SELECT COUNT(*) FROM "LiteLLM_TeamMembership"
               WHERE team_id IN ('member-spend-a', 'member-spend-z')"#,
        )
        .fetch_one(&pool)
        .await?;
        assert_eq!(remaining, 0);
        Ok(())
    }
}
