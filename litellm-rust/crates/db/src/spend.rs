use sqlx::PgExecutor;

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
