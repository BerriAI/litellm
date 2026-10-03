use std::{sync::Arc, time::Duration};

use hmac::{Hmac, Mac};
use litellm_http::Client;
use litellm_storage_clickhouse::READ_LIMITS;
use litellm_traces::QueryScope;
use moka::future::Cache;
use strum::IntoEnumIterator;

use sha2::{Digest, Sha256};
use tokio::sync::{OwnedSemaphorePermit, Semaphore};

use super::{Connection, Error, TraceTable};

const MIB: u64 = 1024 * 1024;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct ReaderLimits {
    pub result_rows: u64,
    pub result_bytes: u64,
    pub memory_bytes: u64,
    pub execution_seconds: u64,
}

impl ReaderLimits {
    pub fn result_mib(&self) -> u64 {
        self.result_bytes / MIB
    }

    pub fn memory_mib(&self) -> u64 {
        self.memory_bytes / MIB
    }
}

pub(crate) const READER_LIMITS: ReaderLimits = ReaderLimits {
    result_rows: READ_LIMITS.result_rows,
    result_bytes: READ_LIMITS.response_bytes as u64,
    memory_bytes: 256 * MIB,
    execution_seconds: READ_LIMITS.execution_seconds,
};

#[derive(Clone)]
pub struct QueryReaders {
    writer: Connection,
    database: String,
    readers: Cache<String, Connection>,
    slots: Arc<Semaphore>,
}

impl QueryReaders {
    pub fn new(writer: Connection, database: String) -> Self {
        Self {
            writer,
            database,
            readers: Cache::builder().max_capacity(1024).build(),
            slots: Arc::new(Semaphore::new(8)),
        }
    }

    pub fn acquire(&self) -> Result<OwnedSemaphorePermit, Error> {
        self.slots
            .clone()
            .try_acquire_owned()
            .map_err(|_| Error::Busy)
    }

    pub async fn connection(
        &self,
        client: &Client,
        scope: &QueryScope,
        secret: &str,
    ) -> Result<Connection, Error> {
        scope.validate().map_err(|_| Error::InvalidScope)?;
        if secret.is_empty() {
            return Err(Error::MissingSecret);
        }
        let identity = serde_json::to_vec(&("litellm_trace_reader_v1", &self.database, scope))
            .map_err(|_| Error::InvalidScope)?;
        let user = format!("litellm_traces_{:x}", Sha256::digest(&identity));
        let password = credential(secret, b"password", &identity)?;
        self.readers
            .try_get_with(
                user.clone(),
                self.provision(client, scope, &user, &password),
            )
            .await
            .map_err(Error::Cached)
    }

    async fn provision(
        &self,
        client: &Client,
        scope: &QueryScope,
        user: &str,
        password: &str,
    ) -> Result<Connection, Error> {
        let database = &self.database;
        if database.is_empty()
            || !database
                .bytes()
                .all(|c| c.is_ascii_alphanumeric() || c == b'_')
        {
            return Err(Error::InvalidScope);
        }
        let password_hash = format!("{:x}", Sha256::digest(password));
        let ReaderLimits {
            result_rows,
            result_bytes,
            memory_bytes,
            execution_seconds,
        } = READER_LIMITS;
        self.execute(
            client,
            format!(
                "CREATE USER IF NOT EXISTS {user} IDENTIFIED WITH sha256_hash BY '{password_hash}' \
             SETTINGS readonly = 1 CONST, max_execution_time = {execution_seconds} CONST, \
             max_result_rows = {result_rows} CONST, max_result_bytes = {result_bytes} CONST, \
             result_overflow_mode = 'throw' CONST, max_memory_usage = {memory_bytes} CONST, \
             max_threads = 2 CONST, max_concurrent_queries_for_user = 8 CONST"
            ),
        )
        .await?;
        self.execute(
            client,
            format!("ALTER USER {user} IDENTIFIED WITH sha256_hash BY '{password_hash}'"),
        )
        .await?;
        for table in TraceTable::iter() {
            let predicate = predicate(scope, table);
            self.execute(
                client,
                format!(
                    "CREATE ROW POLICY IF NOT EXISTS {user}_allow ON `{database}`.{table} \
                 USING 1 TO {user}"
                ),
            )
            .await?;
            self.execute(
                client,
                format!(
                    "CREATE ROW POLICY IF NOT EXISTS {user}_scope ON `{database}`.{table} \
                 AS RESTRICTIVE USING {predicate} TO {user}"
                ),
            )
            .await?;
        }
        for table in TraceTable::iter() {
            self.execute(
                client,
                format!("GRANT SELECT ON `{database}`.{table} TO {user}"),
            )
            .await?;
        }
        Connection::configured(
            &self.writer.url()[..url::Position::AfterPath],
            database,
            user,
            password,
        )
        .map_err(Error::Storage)
    }

    async fn execute(&self, client: &Client, sql: String) -> Result<(), Error> {
        let response = client
            .post(self.writer.url().clone())
            .timeout(Duration::from_secs(15))
            .body(sql)
            .send()
            .await
            .map_err(|_| Error::ProvisionTransport)?;
        if !response.status().is_success() {
            return Err(Error::ProvisionFailed(response.status().as_u16()));
        }
        Ok(())
    }
}

fn predicate(scope: &QueryScope, table: TraceTable) -> String {
    let (team, key) = match table {
        TraceTable::OtelTraces | TraceTable::AgentTracesByKey => ("TeamId", "ApiKeyHash"),
        TraceTable::SpendLogs => ("team_id", "api_key"),
    };
    match scope {
        QueryScope::Admin => "1".to_owned(),
        QueryScope::Logs {
            user_id,
            team_ids,
            api_key_hash,
        } => {
            let owner = literal(user_id);
            let user_clause = match table {
                TraceTable::OtelTraces => format!("UserId = {owner}"),
                TraceTable::AgentTracesByKey => format!("UserIds = [{owner}]"),
                TraceTable::SpendLogs => format!("user = {owner}"),
            };
            let teams = team_ids
                .iter()
                .map(|value| literal(value))
                .collect::<Vec<_>>()
                .join(", ");
            let team_clause = if team_ids.is_empty() {
                "0".to_owned()
            } else {
                format!("{team} IN ({teams})")
            };
            format!(
                "({owner} != '' AND {user_clause}) OR ({team_clause}) OR ({hash} != '' AND {key} = {hash})",
                owner = owner,
                hash = literal(api_key_hash),
            )
        }
        QueryScope::Team { team_id } => format!("{team} = {}", literal(team_id)),
        QueryScope::Key {
            team_id,
            api_key_hash,
        } => format!(
            "{team} = {} AND {key} = {}",
            literal(team_id),
            literal(api_key_hash)
        ),
    }
}

fn credential(secret: &str, purpose: &[u8], identity: &[u8]) -> Result<String, Error> {
    let mut mac =
        Hmac::<Sha256>::new_from_slice(secret.as_bytes()).map_err(|_| Error::MissingSecret)?;
    mac.update(purpose);
    mac.update(identity);
    Ok(format!("{:x}", mac.finalize().into_bytes()))
}

fn literal(value: &str) -> String {
    format!("'{}'", value.replace('\\', "\\\\").replace('\'', "\\'"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::rstest;

    #[rstest]
    #[case::otel(TraceTable::OtelTraces, "TeamId", "ApiKeyHash")]
    #[case::agent(TraceTable::AgentTracesByKey, "TeamId", "ApiKeyHash")]
    #[case::spend(TraceTable::SpendLogs, "team_id", "api_key")]
    fn predicates_preserve_scope_and_escape_values(
        #[case] table: TraceTable,
        #[case] team: &str,
        #[case] key: &str,
    ) {
        assert_eq!(predicate(&QueryScope::Admin, table), "1");
        assert_eq!(
            predicate(
                &QueryScope::Team {
                    team_id: "team'\\".into()
                },
                table
            ),
            format!("{team} = 'team\\'\\\\'")
        );
        assert_eq!(
            predicate(
                &QueryScope::Key {
                    team_id: "".into(),
                    api_key_hash: "key'\\".into()
                },
                table
            ),
            format!("{team} = '' AND {key} = 'key\\'\\\\'")
        );
    }
}
