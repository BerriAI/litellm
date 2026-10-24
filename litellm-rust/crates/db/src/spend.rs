use sqlx::{PgExecutor, PgPool, Postgres, Transaction};

use crate::Error;

#[derive(Clone, Debug, Default, PartialEq)]
pub struct EntitySpend {
    pub ids: Vec<String>,
    pub costs: Vec<f64>,
}

impl EntitySpend {
    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }
}

impl FromIterator<(String, f64)> for EntitySpend {
    fn from_iter<I: IntoIterator<Item = (String, f64)>>(entries: I) -> Self {
        let mut entries: Vec<_> = entries.into_iter().collect();
        entries.sort_by(|left, right| left.0.cmp(&right.0));
        let (ids, costs) = entries.into_iter().unzip();
        Self { ids, costs }
    }
}

pub async fn add_key_spend(
    executor: impl PgExecutor<'_>,
    spend: &EntitySpend,
) -> Result<(), Error> {
    if spend.is_empty() {
        return Ok(());
    }
    sqlx::query_file!("queries/spend/add_key_spend.sql", &spend.ids, &spend.costs)
        .execute(executor)
        .await?;
    Ok(())
}
pub async fn add_user_spend(
    executor: impl PgExecutor<'_>,
    spend: &EntitySpend,
) -> Result<(), Error> {
    if spend.is_empty() {
        return Ok(());
    }
    sqlx::query_file!("queries/spend/add_user_spend.sql", &spend.ids, &spend.costs)
        .execute(executor)
        .await?;
    Ok(())
}

pub async fn add_team_spend(
    executor: impl PgExecutor<'_>,
    spend: &EntitySpend,
) -> Result<(), Error> {
    if spend.is_empty() {
        return Ok(());
    }
    sqlx::query_file!("queries/spend/add_team_spend.sql", &spend.ids, &spend.costs)
        .execute(executor)
        .await?;
    Ok(())
}

pub async fn add_end_user_spend(
    executor: impl PgExecutor<'_>,
    spend: &EntitySpend,
) -> Result<(), Error> {
    if spend.is_empty() {
        return Ok(());
    }
    sqlx::query_file!(
        "queries/spend/add_end_user_spend.sql",
        &spend.ids,
        &spend.costs
    )
    .execute(executor)
    .await?;
    Ok(())
}

pub async fn begin(pool: &PgPool) -> Result<Transaction<'static, Postgres>, Error> {
    let mut transaction = pool.begin().await?;
    sqlx::query!("SELECT set_config('lock_timeout', '5000ms', true)")
        .fetch_one(&mut *transaction)
        .await?;
    Ok(transaction)
}

pub async fn lock_team(
    transaction: &mut Transaction<'_, Postgres>,
    team_id: &str,
) -> Result<(), Error> {
    sqlx::query_file!("queries/spend/lock_team.sql", team_id)
        .fetch_one(&mut **transaction)
        .await?;
    Ok(())
}

#[derive(Clone, Debug, PartialEq)]
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
