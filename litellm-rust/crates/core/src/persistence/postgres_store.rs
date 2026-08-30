use serde_json::Value;
use sqlx::PgPool;
use sqlx::postgres::PgPoolOptions;

use super::{DatabaseStore, PersistenceError};

/// PostgreSQL-backed database store.
///
/// Uses `sqlx::PgPool` for connection pooling and async queries.
pub struct PostgresStore {
    pool: PgPool,
}

impl PostgresStore {
    /// Connect to PostgreSQL at the given URL.
    ///
    /// URL format: `postgres://user:password@host:port/database`
    pub async fn connect(url: &str) -> Result<Self, PersistenceError> {
        let pool = PgPoolOptions::new()
            .max_connections(10)
            .connect(url)
            .await
            .map_err(|e| {
                PersistenceError::Connection(format!("Postgres connection failed: {e}"))
            })?;
        Ok(Self { pool })
    }

    /// Create from an existing pool (for testing/sharing).
    pub fn from_pool(pool: PgPool) -> Self {
        Self { pool }
    }

    /// Clone the underlying pool (for sharing across components).
    pub fn clone_pool(&self) -> PgPool {
        self.pool.clone()
    }

    /// Get a reference to the underlying pool.
    pub fn pool(&self) -> &PgPool {
        &self.pool
    }

    /// Ping PostgreSQL to verify connectivity.
    pub async fn ping(&self) -> Result<(), PersistenceError> {
        sqlx::query("SELECT 1")
            .execute(&self.pool)
            .await
            .map_err(|e| PersistenceError::Postgres(format!("ping failed: {e}")))?;
        Ok(())
    }
}

impl DatabaseStore for PostgresStore {
    async fn insert_spend_log(&self, log: &Value) -> Result<(), PersistenceError> {
        let request_id = log.get("request_id").and_then(|v| v.as_str()).unwrap_or("");
        let call_type = log.get("call_type").and_then(|v| v.as_str()).unwrap_or("");
        let api_key = log.get("api_key").and_then(|v| v.as_str()).unwrap_or("");
        let spend = log.get("spend").and_then(|v| v.as_f64()).unwrap_or(0.0);
        let total_tokens = log
            .get("total_tokens")
            .and_then(|v| v.as_i64())
            .unwrap_or(0);
        let prompt_tokens = log
            .get("prompt_tokens")
            .and_then(|v| v.as_i64())
            .unwrap_or(0);
        let completion_tokens = log
            .get("completion_tokens")
            .and_then(|v| v.as_i64())
            .unwrap_or(0);
        let model = log.get("model").and_then(|v| v.as_str()).unwrap_or("");
        let user = log.get("user").and_then(|v| v.as_str());
        let team_id = log.get("team_id").and_then(|v| v.as_str());
        let organization_id = log.get("organization_id").and_then(|v| v.as_str());

        sqlx::query(
            r#"
            INSERT INTO "LiteLLM_SpendLogs" (
                request_id, call_type, api_key, spend,
                total_tokens, prompt_tokens, completion_tokens,
                model, "user", team_id, organization_id,
                metadata, "startTime", "endTime", status
            ) VALUES (
                $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11,
                $12, NOW(), NOW(), $13
            )
            ON CONFLICT (request_id) DO NOTHING
            "#,
        )
        .bind(request_id)
        .bind(call_type)
        .bind(api_key)
        .bind(spend)
        .bind(total_tokens)
        .bind(prompt_tokens)
        .bind(completion_tokens)
        .bind(model)
        .bind(user)
        .bind(team_id)
        .bind(organization_id)
        .bind(log)
        .bind(
            log.get("status")
                .and_then(|v| v.as_str())
                .unwrap_or("success"),
        )
        .execute(&self.pool)
        .await
        .map_err(|e| PersistenceError::Postgres(format!("INSERT spend log failed: {e}")))?;

        Ok(())
    }

    async fn batch_insert_spend_logs(&self, logs: &[Value]) -> Result<(), PersistenceError> {
        for log in logs {
            self.insert_spend_log(log).await?;
        }
        Ok(())
    }

    async fn update_entity_spend(
        &self,
        entity_type: &str,
        entity_id: &str,
        amount: f64,
    ) -> Result<(), PersistenceError> {
        let (table, id_column) = match entity_type {
            "key" => ("\"LiteLLM_VerificationToken\"", "token"),
            "user" => ("\"LiteLLM_UserTable\"", "user_id"),
            "team" => ("\"LiteLLM_TeamTable\"", "team_id"),
            "organization" => ("\"LiteLLM_OrganizationTable\"", "org_id"),
            "end_user" => ("\"LiteLLM_EndUserTable\"", "user_id"),
            "tag" => ("\"LiteLLM_TagTable\"", "tag_name"),
            "agent" => ("\"LiteLLM_AgentsTable\"", "agent_id"),
            "team_member" => ("\"LiteLLM_TeamMembership\"", "user_id"),
            _ => {
                return Err(PersistenceError::Postgres(format!(
                    "unknown entity type: {entity_type}"
                )));
            }
        };

        let query = format!(
            r#"UPDATE {} SET spend = spend + $1 WHERE {} = $2"#,
            table, id_column
        );

        sqlx::query(&query)
            .bind(amount)
            .bind(entity_id)
            .execute(&self.pool)
            .await
            .map_err(|e| {
                PersistenceError::Postgres(format!(
                    "UPDATE spend for {entity_type}/{entity_id} failed: {e}"
                ))
            })?;

        Ok(())
    }

    async fn batch_update_entity_spend(
        &self,
        updates: &[(String, String, f64)],
    ) -> Result<(), PersistenceError> {
        for (entity_type, entity_id, amount) in updates {
            self.update_entity_spend(entity_type, entity_id, *amount)
                .await?;
        }
        Ok(())
    }
}

impl PostgresStore {
    /// Get key information by hashed token.
    pub async fn get_key_by_hashed_token(&self, hashed_token: &str) -> Result<Option<Value>, PersistenceError> {
        let row: Option<(Value,)> = sqlx::query_as(
            r#"
            SELECT row_to_json(t)
            FROM (
                SELECT token, key_name, key_alias, user_id, team_id, org_id,
                       max_budget, budget_duration, spend, models,
                       tpm_limit, rpm_limit, allowed_routes, metadata
                FROM "LiteLLM_VerificationToken"
                WHERE token = $1
            ) t
            "#,
        )
        .bind(hashed_token)
        .fetch_optional(&self.pool)
        .await
        .map_err(|e| PersistenceError::Postgres(format!("get_key_by_hashed_token failed: {e}")))?;

        Ok(row.map(|r| r.0))
    }

    /// Get spend logs with optional filtering and pagination.
    pub async fn get_spend_logs(
        &self,
        start_time: Option<&str>,
        end_time: Option<&str>,
        user_id: Option<&str>,
        team_id: Option<&str>,
        model: Option<&str>,
        limit: i64,
        offset: i64,
    ) -> Result<Vec<Value>, PersistenceError> {
        // Build query with all possible filters
        let rows: Vec<(Value,)> = if let (Some(start), Some(end), Some(user), Some(team), Some(m)) =
            (start_time, end_time, user_id, team_id, model)
        {
            sqlx::query_as(
                r#"
                SELECT row_to_json(t)
                FROM (
                    SELECT request_id, call_type, api_key, spend,
                           total_tokens, prompt_tokens, completion_tokens,
                           model, "user", team_id, organization_id,
                           metadata, "startTime", "endTime", status
                    FROM "LiteLLM_SpendLogs"
                    WHERE "startTime" >= $1 AND "endTime" <= $2
                      AND "user" = $3 AND team_id = $4 AND model = $5
                    ORDER BY "startTime" DESC
                    LIMIT $6 OFFSET $7
                ) t
                "#,
            )
            .bind(start)
            .bind(end)
            .bind(user)
            .bind(team)
            .bind(m)
            .bind(limit)
            .bind(offset)
            .fetch_all(&self.pool)
            .await
            .map_err(|e| PersistenceError::Postgres(format!("get_spend_logs failed: {e}")))?
        } else if let (Some(start), Some(end)) = (start_time, end_time) {
            sqlx::query_as(
                r#"
                SELECT row_to_json(t)
                FROM (
                    SELECT request_id, call_type, api_key, spend,
                           total_tokens, prompt_tokens, completion_tokens,
                           model, "user", team_id, organization_id,
                           metadata, "startTime", "endTime", status
                    FROM "LiteLLM_SpendLogs"
                    WHERE "startTime" >= $1 AND "endTime" <= $2
                    ORDER BY "startTime" DESC
                    LIMIT $3 OFFSET $4
                ) t
                "#,
            )
            .bind(start)
            .bind(end)
            .bind(limit)
            .bind(offset)
            .fetch_all(&self.pool)
            .await
            .map_err(|e| PersistenceError::Postgres(format!("get_spend_logs failed: {e}")))?
        } else {
            sqlx::query_as(
                r#"
                SELECT row_to_json(t)
                FROM (
                    SELECT request_id, call_type, api_key, spend,
                           total_tokens, prompt_tokens, completion_tokens,
                           model, "user", team_id, organization_id,
                           metadata, "startTime", "endTime", status
                    FROM "LiteLLM_SpendLogs"
                    ORDER BY "startTime" DESC
                    LIMIT $1 OFFSET $2
                ) t
                "#,
            )
            .bind(limit)
            .bind(offset)
            .fetch_all(&self.pool)
            .await
            .map_err(|e| PersistenceError::Postgres(format!("get_spend_logs failed: {e}")))?
        };

        Ok(rows.into_iter().map(|r| r.0).collect())
    }

    /// Get user information by user ID.
    pub async fn get_user_by_id(&self, user_id: &str) -> Result<Option<Value>, PersistenceError> {
        let row: Option<(Value,)> = sqlx::query_as(
            r#"
            SELECT row_to_json(t)
            FROM (
                SELECT user_id, user_email, user_role, max_budget,
                       budget_duration, spend, models, tpm_limit, rpm_limit,
                       metadata, "createdAt", "updatedAt"
                FROM "LiteLLM_UserTable"
                WHERE user_id = $1
            ) t
            "#,
        )
        .bind(user_id)
        .fetch_optional(&self.pool)
        .await
        .map_err(|e| PersistenceError::Postgres(format!("get_user_by_id failed: {e}")))?;

        Ok(row.map(|r| r.0))
    }

    /// Get team information by team ID.
    pub async fn get_team_by_id(&self, team_id: &str) -> Result<Option<Value>, PersistenceError> {
        let row: Option<(Value,)> = sqlx::query_as(
            r#"
            SELECT row_to_json(t)
            FROM (
                SELECT team_id, team_alias, organization_id, max_budget,
                       budget_duration, spend, models, tpm_limit, rpm_limit,
                       metadata, "createdAt", "updatedAt"
                FROM "LiteLLM_TeamTable"
                WHERE team_id = $1
            ) t
            "#,
        )
        .bind(team_id)
        .fetch_optional(&self.pool)
        .await
        .map_err(|e| PersistenceError::Postgres(format!("get_team_by_id failed: {e}")))?;

        Ok(row.map(|r| r.0))
    }

    /// Get organization information by organization ID.
    pub async fn get_organization_by_id(&self, org_id: &str) -> Result<Option<Value>, PersistenceError> {
        let row: Option<(Value,)> = sqlx::query_as(
            r#"
            SELECT row_to_json(t)
            FROM (
                SELECT org_id, organization_alias, max_budget,
                       budget_duration, spend, models, tpm_limit, rpm_limit,
                       metadata, "createdAt", "updatedAt"
                FROM "LiteLLM_OrganizationTable"
                WHERE org_id = $1
            ) t
            "#,
        )
        .bind(org_id)
        .fetch_optional(&self.pool)
        .await
        .map_err(|e| PersistenceError::Postgres(format!("get_organization_by_id failed: {e}")))?;

        Ok(row.map(|r| r.0))
    }
}
