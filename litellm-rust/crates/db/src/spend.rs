//! The proxy's spend writes: additive spend on the key, user, team, team membership and end user
//! rows, rows in `LiteLLM_SpendLogs`, and the daily activity tables. Every statement takes a
//! batch, so a flush costs one round trip per table.

use sqlx::{PgPool, Postgres, Transaction};

use crate::Error;

/// Spend to add per id. Ids are locked in sorted order, as the proxy does, so writers on other
/// replicas cannot deadlock with this one.
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

#[derive(Clone, Debug, PartialEq)]
pub struct TeamMemberSpend {
    pub team_id: String,
    pub user_id: String,
    pub cost: f64,
}

/// Starts a transaction that waits at most 5 s for a row lock, the proxy's
/// `SPEND_ROLLUP_LOCK_TIMEOUT_MS`.
async fn begin(pool: &PgPool) -> Result<Transaction<'static, Postgres>, Error> {
    let mut transaction = pool.begin().await?;
    sqlx::query!("SELECT set_config('lock_timeout', '5000ms', true)")
        .fetch_one(&mut *transaction)
        .await?;
    Ok(transaction)
}

/// `LiteLLM_VerificationToken.spend` and `total_spend`, and `last_active` set to now.
pub async fn add_key_spend(pool: &PgPool, spend: &EntitySpend) -> Result<(), Error> {
    if spend.is_empty() {
        return Ok(());
    }
    let mut transaction = begin(pool).await?;
    sqlx::query!(
        r#"WITH added AS (SELECT * FROM UNNEST($1::text[], $2::float8[]) AS a(id, cost)),
                locked AS (SELECT t.token FROM "LiteLLM_VerificationToken" t
                           WHERE t.token IN (SELECT id FROM added) ORDER BY t.token FOR UPDATE)
           UPDATE "LiteLLM_VerificationToken" t
           SET spend = t.spend + added.cost,
               total_spend = t.total_spend + added.cost,
               last_active = (NOW() AT TIME ZONE 'UTC'),
               updated_at = (NOW() AT TIME ZONE 'UTC')
           FROM added
           WHERE t.token = added.id AND t.token IN (SELECT token FROM locked)"#,
        &spend.ids,
        &spend.costs,
    )
    .execute(&mut *transaction)
    .await?;
    transaction.commit().await?;
    Ok(())
}

/// `LiteLLM_UserTable.spend`. A user without a row is skipped, as the proxy skips it.
pub async fn add_user_spend(pool: &PgPool, spend: &EntitySpend) -> Result<(), Error> {
    if spend.is_empty() {
        return Ok(());
    }
    let mut transaction = begin(pool).await?;
    sqlx::query!(
        r#"WITH added AS (SELECT * FROM UNNEST($1::text[], $2::float8[]) AS a(id, cost)),
                locked AS (SELECT u.user_id FROM "LiteLLM_UserTable" u
                           WHERE u.user_id IN (SELECT id FROM added) ORDER BY u.user_id FOR UPDATE)
           UPDATE "LiteLLM_UserTable" u
           SET spend = u.spend + added.cost, updated_at = (NOW() AT TIME ZONE 'UTC')
           FROM added
           WHERE u.user_id = added.id AND u.user_id IN (SELECT user_id FROM locked)"#,
        &spend.ids,
        &spend.costs,
    )
    .execute(&mut *transaction)
    .await?;
    transaction.commit().await?;
    Ok(())
}

/// `LiteLLM_TeamTable.spend`.
pub async fn add_team_spend(pool: &PgPool, spend: &EntitySpend) -> Result<(), Error> {
    if spend.is_empty() {
        return Ok(());
    }
    let mut transaction = begin(pool).await?;
    sqlx::query!(
        r#"WITH added AS (SELECT * FROM UNNEST($1::text[], $2::float8[]) AS a(id, cost)),
                locked AS (SELECT t.team_id FROM "LiteLLM_TeamTable" t
                           WHERE t.team_id IN (SELECT id FROM added) ORDER BY t.team_id FOR UPDATE)
           UPDATE "LiteLLM_TeamTable" t
           SET spend = t.spend + added.cost, updated_at = (NOW() AT TIME ZONE 'UTC')
           FROM added
           WHERE t.team_id = added.id AND t.team_id IN (SELECT team_id FROM locked)"#,
        &spend.ids,
        &spend.costs,
    )
    .execute(&mut *transaction)
    .await?;
    transaction.commit().await?;
    Ok(())
}

/// `LiteLLM_EndUserTable.spend`, creating the end user on first spend as the proxy does.
pub async fn add_end_user_spend(pool: &PgPool, spend: &EntitySpend) -> Result<(), Error> {
    if spend.is_empty() {
        return Ok(());
    }
    let mut transaction = begin(pool).await?;
    sqlx::query!(
        r#"INSERT INTO "LiteLLM_EndUserTable" (user_id, spend, blocked)
           SELECT id, cost, false FROM UNNEST($1::text[], $2::float8[]) AS a(id, cost) ORDER BY id
           ON CONFLICT (user_id) DO UPDATE
           SET spend = "LiteLLM_EndUserTable".spend + EXCLUDED.spend"#,
        &spend.ids,
        &spend.costs,
    )
    .execute(&mut *transaction)
    .await?;
    transaction.commit().await?;
    Ok(())
}

/// `LiteLLM_TeamMembership.spend` and `total_spend`. A membership row is created only for a
/// user on the team's roster, under a per-team advisory lock, as the proxy writes it.
pub async fn add_team_member_spend(
    pool: &PgPool,
    members: &[TeamMemberSpend],
) -> Result<(), Error> {
    if members.is_empty() {
        return Ok(());
    }
    let mut members = members.to_vec();
    members.sort_by(|left, right| {
        (&left.team_id, &left.user_id).cmp(&(&right.team_id, &right.user_id))
    });
    let mut transaction = begin(pool).await?;
    let mut teams: Vec<&str> = members
        .iter()
        .map(|member| member.team_id.as_str())
        .collect();
    teams.dedup();
    for team in teams {
        sqlx::query!(
            r#"SELECT pg_advisory_xact_lock(hashtext($1)) IS NULL AS "locked!""#,
            team
        )
        .fetch_one(&mut *transaction)
        .await?;
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
    sqlx::query!(
        r#"INSERT INTO "LiteLLM_TeamMembership" (user_id, team_id, spend, total_spend)
           SELECT member.user_id, member.team_id, member.cost, member.cost
           FROM jsonb_to_recordset($1::jsonb) AS member(user_id text, team_id text, cost float8)
           WHERE EXISTS (SELECT 1 FROM "LiteLLM_TeamTable" t WHERE t.team_id = member.team_id
                 AND t.members_with_roles @> jsonb_build_array(jsonb_build_object('user_id', member.user_id)))
              OR EXISTS (SELECT 1 FROM "LiteLLM_TeamMembership" m
                 WHERE m.user_id = member.user_id AND m.team_id = member.team_id)
           ON CONFLICT (user_id, team_id) DO UPDATE
           SET spend = "LiteLLM_TeamMembership".spend + EXCLUDED.spend,
               total_spend = "LiteLLM_TeamMembership".total_spend + EXCLUDED.total_spend"#,
        rows,
    )
    .execute(&mut *transaction)
    .await?;
    transaction.commit().await?;
    Ok(())
}

/// One `LiteLLM_SpendLogs` row. Times are UTC, `YYYY-MM-DD HH:MM:SS.fff`.
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

/// Inserts spend logs, skipping request ids already written, as the proxy's
/// `create_many(skip_duplicates=True)` does.
pub async fn insert_spend_logs(pool: &PgPool, logs: &[SpendLog]) -> Result<(), Error> {
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
    sqlx::query!(
        r#"INSERT INTO "LiteLLM_SpendLogs" (
               request_id, call_type, api_key, spend, total_tokens, prompt_tokens,
               completion_tokens, "startTime", "endTime", "completionStartTime",
               request_duration_ms, model, model_id, model_group, custom_llm_provider, api_base,
               "user", metadata, cache_hit, cache_key, request_tags, team_id, organization_id,
               end_user, requester_ip_address, messages, response, proxy_server_request,
               session_id, status, litellm_call_id)
           SELECT l.request_id, l.call_type, l.api_key, l.spend, l.total_tokens, l.prompt_tokens,
                  l.completion_tokens, l.start_time::timestamp(3), l.end_time::timestamp(3),
                  l.completion_start_time::timestamp(3), l.request_duration_ms, l.model,
                  l.model_id, l.model_group, l.custom_llm_provider, l.api_base, l.user_id,
                  l.metadata, l.cache_hit, l.cache_key, '[]'::jsonb, l.team_id,
                  l.organization_id, l.end_user, l.requester_ip_address, '{}'::jsonb,
                  '{}'::jsonb, '{}'::jsonb, l.session_id, l.status, l.litellm_call_id
           FROM UNNEST($1::text[], $2::text[], $3::text[], $4::float8[], $5::int4[], $6::int4[],
                       $7::int4[], $8::text[], $9::text[], $10::text[], $11::int4[], $12::text[],
                       $13::text[], $14::text[], $15::text[], $16::text[], $17::text[],
                       $18::jsonb[], $19::text[], $20::text[], $21::text[], $22::text[],
                       $23::text[], $24::text[], $25::text[], $26::text[], $27::text[])
                AS l(request_id, call_type, api_key, spend, total_tokens, prompt_tokens,
                     completion_tokens, start_time, end_time, completion_start_time,
                     request_duration_ms, model, model_id, model_group, custom_llm_provider,
                     api_base, user_id, metadata, cache_hit, cache_key, team_id, organization_id,
                     end_user, requester_ip_address, session_id, status, litellm_call_id)
           ON CONFLICT (request_id) DO NOTHING"#,
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
    .execute(pool)
    .await?;
    Ok(())
}

/// The daily activity tables, keyed by user, team or end user.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum DailyTable {
    User,
    Team,
    EndUser,
}

/// One row of a daily activity table. Key columns are `''` rather than NULL, as the proxy
/// writes them, so its unique index matches.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct DailySpend {
    pub entity_id: String,
    /// `YYYY-MM-DD`, UTC.
    pub date: String,
    pub api_key: String,
    pub model: String,
    pub custom_llm_provider: String,
    pub endpoint: String,
    /// Written on insert only.
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

/// Adds `rows` to a daily table. Rows must be unique by their key columns.
pub async fn upsert_daily_spend(
    pool: &PgPool,
    table: DailyTable,
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
    let entity_id = column!(entity_id);
    let date = column!(date);
    let api_key = column!(api_key);
    let model = column!(model);
    let custom_llm_provider = column!(custom_llm_provider);
    let endpoint = column!(endpoint);
    let model_group: Vec<Option<String>> = column!(model_group);
    let prompt_tokens = column!(prompt_tokens);
    let completion_tokens = column!(completion_tokens);
    let api_requests = column!(api_requests);
    let successful_requests = column!(successful_requests);
    let failed_requests = column!(failed_requests);
    let cache_read = column!(cache_read_input_tokens);
    let cache_creation = column!(cache_creation_input_tokens);
    let response_time = column!(total_response_time_ms);
    let timed_requests = column!(timed_requests);
    let spend = column!(spend);
    let mut transaction = begin(pool).await?;
    // One statement per table: a table name cannot be a parameter.
    macro_rules! upsert {
        ($table:literal, $entity:literal) => {
            sqlx::query!(
                r#"INSERT INTO "# + $table + r#" (id, "# + $entity + r#", date, api_key, model,
                       custom_llm_provider, mcp_namespaced_tool_name, endpoint, model_group,
                       prompt_tokens, completion_tokens, api_requests, successful_requests,
                       failed_requests, cache_read_input_tokens, cache_creation_input_tokens,
                       total_response_time_ms, timed_requests, spend, updated_at)
                   SELECT gen_random_uuid()::text, r.entity_id, r.date, r.api_key, r.model,
                          r.custom_llm_provider, '', r.endpoint, r.model_group, r.prompt_tokens,
                          r.completion_tokens, r.api_requests, r.successful_requests,
                          r.failed_requests, r.cache_read, r.cache_creation, r.response_time,
                          r.timed_requests, r.spend, (NOW() AT TIME ZONE 'UTC')
                   FROM UNNEST($1::text[], $2::text[], $3::text[], $4::text[], $5::text[],
                               $6::text[], $7::text[], $8::int8[], $9::int8[], $10::int8[],
                               $11::int8[], $12::int8[], $13::int8[], $14::int8[], $15::int8[],
                               $16::int8[], $17::float8[])
                        AS r(entity_id, date, api_key, model, custom_llm_provider, endpoint,
                             model_group, prompt_tokens, completion_tokens, api_requests,
                             successful_requests, failed_requests, cache_read, cache_creation,
                             response_time, timed_requests, spend)
                   ORDER BY r.date, r.entity_id, r.api_key, r.model, r.custom_llm_provider
                   ON CONFLICT ("# + $entity + r#", date, api_key, model, custom_llm_provider,
                                mcp_namespaced_tool_name, endpoint) DO UPDATE SET
                       prompt_tokens = "# + $table + r#".prompt_tokens + EXCLUDED.prompt_tokens,
                       completion_tokens = "# + $table + r#".completion_tokens + EXCLUDED.completion_tokens,
                       api_requests = "# + $table + r#".api_requests + EXCLUDED.api_requests,
                       successful_requests = "# + $table + r#".successful_requests + EXCLUDED.successful_requests,
                       failed_requests = "# + $table + r#".failed_requests + EXCLUDED.failed_requests,
                       cache_read_input_tokens = "# + $table + r#".cache_read_input_tokens + EXCLUDED.cache_read_input_tokens,
                       cache_creation_input_tokens = "# + $table + r#".cache_creation_input_tokens + EXCLUDED.cache_creation_input_tokens,
                       total_response_time_ms = "# + $table + r#".total_response_time_ms + EXCLUDED.total_response_time_ms,
                       timed_requests = "# + $table + r#".timed_requests + EXCLUDED.timed_requests,
                       spend = "# + $table + r#".spend + EXCLUDED.spend,
                       updated_at = (NOW() AT TIME ZONE 'UTC')"#,
                &entity_id,
                &date,
                &api_key,
                &model,
                &custom_llm_provider,
                &endpoint,
                &model_group as &[Option<String>],
                &prompt_tokens,
                &completion_tokens,
                &api_requests,
                &successful_requests,
                &failed_requests,
                &cache_read,
                &cache_creation,
                &response_time,
                &timed_requests,
                &spend,
            )
            .execute(&mut *transaction)
            .await?
        };
    }
    match table {
        DailyTable::User => upsert!(r#""LiteLLM_DailyUserSpend""#, "user_id"),
        DailyTable::Team => upsert!(r#""LiteLLM_DailyTeamSpend""#, "team_id"),
        DailyTable::EndUser => upsert!(r#""LiteLLM_DailyEndUserSpend""#, "end_user_id"),
    };
    transaction.commit().await?;
    Ok(())
}

impl Error {
    /// Whether retrying the same write can succeed: a deadlock, a lock wait timeout, a
    /// serialization failure, or a lost connection.
    pub fn is_retryable(&self) -> bool {
        match self {
            Self::Query(sqlx::Error::Database(error)) => {
                matches!(error.code().as_deref(), Some("40P01" | "55P03" | "40001"))
            }
            Self::Query(
                sqlx::Error::Io(_) | sqlx::Error::PoolTimedOut | sqlx::Error::PoolClosed,
            ) => true,
            Self::Query(_) | Self::Url(_) => false,
        }
    }
}
