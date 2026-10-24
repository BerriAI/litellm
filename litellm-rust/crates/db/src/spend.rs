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
