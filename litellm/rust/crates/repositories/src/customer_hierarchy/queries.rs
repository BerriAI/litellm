use serde::Deserialize;
use sqlx::{
    PgExecutor,
    types::{Json, chrono::NaiveDateTime},
};

use crate::Budget;

#[derive(Debug, Deserialize)]
pub struct CustomerOrgRow {
    pub customer_org_id: String,
    pub alias: Option<String>,
    pub spend: f64,
    pub budget: Option<Budget>,
}

#[derive(Debug, Deserialize)]
pub struct CustomerTeamRow {
    pub customer_team_id: String,
    pub alias: Option<String>,
    pub spend: f64,
    pub budget: Option<Budget>,
    pub budget_override: Option<Budget>,
    pub budget_override_expires_at: Option<NaiveDateTime>,
}

#[derive(Debug, Deserialize)]
pub struct CustomerMemberRow {
    pub customer_member_id: String,
    pub alias: Option<String>,
    pub spend: f64,
    pub budget: Option<Budget>,
    pub budget_override: Option<Budget>,
    pub budget_override_expires_at: Option<NaiveDateTime>,
    pub parent_org: CustomerOrgRow,
    pub teams: Vec<CustomerTeamRow>,
}

pub(super) async fn member(
    executor: impl PgExecutor<'_>,
    id: &str,
) -> Result<Option<CustomerMemberRow>, litellm_db::Error> {
    Ok(
        sqlx::query_file_scalar!("src/customer_hierarchy/queries/member.sql", id)
            .fetch_optional(executor)
            .await?
            .map(|Json(row)| row),
    )
}

pub(super) async fn organizations(
    executor: impl PgExecutor<'_>,
    alias: &str,
) -> Result<Vec<CustomerOrgRow>, litellm_db::Error> {
    Ok(
        sqlx::query_file_scalar!("src/customer_hierarchy/queries/organizations.sql", alias)
            .fetch_all(executor)
            .await?
            .into_iter()
            .map(|Json(row)| row)
            .collect(),
    )
}
