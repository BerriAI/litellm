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
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SpendLog {
    pub request_id: String,
    pub call_type: String,
    pub api_key: String,
    pub spend: f64,
    pub total_tokens: i32,
    pub prompt_tokens: i32,
    pub completion_tokens: i32,
    pub start_time: String,
    pub end_time: String,
    pub completion_start_time: Option<String>,
    pub request_duration_ms: Option<i32>,
    pub model: String,
    pub model_id: String,
    pub model_group: String,
    pub custom_llm_provider: String,
    pub api_base: String,
    pub user: String,
    pub metadata: serde_json::Value,
    pub cache_hit: String,
    pub cache_key: String,
    pub team_id: String,
    pub organization_id: String,
    pub end_user: String,
    pub requester_ip_address: Option<String>,
    pub session_id: Option<String>,
    pub status: String,
    pub litellm_call_id: Option<String>,
}

pub async fn insert_spend_logs(
    executor: impl PgExecutor<'_>,
    logs: &[SpendLog],
) -> Result<(), Error> {
    if logs.is_empty() {
        return Ok(());
    }
    macro_rules! column {
        ($field:ident) => {
            logs.iter()
                .map(|log| log.$field.clone())
                .collect::<Vec<_>>()
        };
    }
    sqlx::query_file!(
        "queries/spend/insert_spend_logs.sql",
        &column!(request_id),
        &column!(call_type),
        &column!(api_key),
        &column!(spend),
        &column!(total_tokens),
        &column!(prompt_tokens),
        &column!(completion_tokens),
        &column!(start_time),
        &column!(end_time),
        &column!(completion_start_time) as &[Option<String>],
        &column!(request_duration_ms) as &[Option<i32>],
        &column!(model),
        &column!(model_id),
        &column!(model_group),
        &column!(custom_llm_provider),
        &column!(api_base),
        &column!(user),
        &column!(metadata),
        &column!(cache_hit),
        &column!(cache_key),
        &column!(team_id),
        &column!(organization_id),
        &column!(end_user),
        &column!(requester_ip_address) as &[Option<String>],
        &column!(session_id) as &[Option<String>],
        &column!(status),
        &column!(litellm_call_id) as &[Option<String>],
    )
    .execute(executor)
    .await?;
    Ok(())
}
#[derive(Clone, Debug, Default, PartialEq)]
pub struct DailySpend {
    pub entity_id: String,
    pub date: String,
    pub api_key: String,
    pub model: String,
    pub custom_llm_provider: String,
    pub endpoint: String,
    pub model_group: Option<String>,
    pub prompt_tokens: i64,
    pub completion_tokens: i64,
    pub api_requests: i64,
    pub successful_requests: i64,
    pub failed_requests: i64,
    pub cache_read_input_tokens: i64,
    pub cache_creation_input_tokens: i64,
    pub total_response_time_ms: i64,
    pub timed_requests: i64,
    pub spend: f64,
}

pub async fn upsert_daily_user_spend(
    executor: impl PgExecutor<'_>,
    rows: &[DailySpend],
) -> Result<(), Error> {
    if rows.is_empty() {
        return Ok(());
    }
    macro_rules! column {
        ($field:ident) => {
            rows.iter()
                .map(|row| row.$field.clone())
                .collect::<Vec<_>>()
        };
    }
    sqlx::query_file!(
        "queries/spend/upsert_daily_user_spend.sql",
        &column!(entity_id),
        &column!(date),
        &column!(api_key),
        &column!(model),
        &column!(custom_llm_provider),
        &column!(endpoint),
        &column!(model_group) as &[Option<String>],
        &column!(prompt_tokens),
        &column!(completion_tokens),
        &column!(api_requests),
        &column!(successful_requests),
        &column!(failed_requests),
        &column!(cache_read_input_tokens),
        &column!(cache_creation_input_tokens),
        &column!(total_response_time_ms),
        &column!(timed_requests),
        &column!(spend),
    )
    .execute(executor)
    .await?;
    Ok(())
}

pub async fn upsert_daily_team_spend(
    executor: impl PgExecutor<'_>,
    rows: &[DailySpend],
) -> Result<(), Error> {
    if rows.is_empty() {
        return Ok(());
    }
    macro_rules! column {
        ($field:ident) => {
            rows.iter()
                .map(|row| row.$field.clone())
                .collect::<Vec<_>>()
        };
    }
    sqlx::query_file!(
        "queries/spend/upsert_daily_team_spend.sql",
        &column!(entity_id),
        &column!(date),
        &column!(api_key),
        &column!(model),
        &column!(custom_llm_provider),
        &column!(endpoint),
        &column!(model_group) as &[Option<String>],
        &column!(prompt_tokens),
        &column!(completion_tokens),
        &column!(api_requests),
        &column!(successful_requests),
        &column!(failed_requests),
        &column!(cache_read_input_tokens),
        &column!(cache_creation_input_tokens),
        &column!(total_response_time_ms),
        &column!(timed_requests),
        &column!(spend),
    )
    .execute(executor)
    .await?;
    Ok(())
}
