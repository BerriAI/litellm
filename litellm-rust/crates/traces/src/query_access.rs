use std::{sync::Arc, time::Duration};

use hmac::{Hmac, Mac};
use litellm_http::Client;
use moka::future::Cache;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tokio::sync::{OwnedSemaphorePermit, Semaphore};

use crate::{Connection, QueryAccessError};

const TABLES: [&str; 3] = ["otel_traces", "agent_traces_by_key", "spend_logs"];

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum QueryScope {
    Admin,
    Team {
        team_id: String,
    },
    Key {
        team_id: String,
        api_key_hash: String,
    },
}

impl QueryScope {
    fn validate(&self) -> Result<(), QueryAccessError> {
        match self {
            Self::Admin => Ok(()),
            Self::Team { team_id } if !team_id.is_empty() => Ok(()),
            Self::Key { api_key_hash, .. } if !api_key_hash.is_empty() => Ok(()),
            _ => Err(QueryAccessError::InvalidScope),
        }
    }

    fn predicate(&self, table: &str) -> String {
        let (team, key) = if table == "spend_logs" {
            ("team_id", "api_key")
        } else {
            ("TeamId", "ApiKeyHash")
        };
        match self {
            Self::Admin => "1".to_owned(),
            Self::Team { team_id } => format!("{team} = {}", literal(team_id)),
            Self::Key {
                team_id,
                api_key_hash,
            } => format!(
                "{team} = {} AND {key} = {}",
                literal(team_id),
                literal(api_key_hash)
            ),
        }
    }
}

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

    pub fn acquire(&self) -> Result<OwnedSemaphorePermit, QueryAccessError> {
        self.slots
            .clone()
            .try_acquire_owned()
            .map_err(|_| QueryAccessError::Busy)
    }

    pub async fn connection(
        &self,
        client: &Client,
        scope: &QueryScope,
        secret: &str,
    ) -> Result<Connection, QueryAccessError> {
        scope.validate()?;
        if secret.is_empty() {
            return Err(QueryAccessError::MissingSecret);
        }
        let identity = serde_json::to_vec(&("litellm_trace_reader_v1", &self.database, scope))
            .map_err(|_| QueryAccessError::InvalidScope)?;
        let user = format!("litellm_traces_{:x}", Sha256::digest(&identity));
        let password = credential(secret, b"password", &identity)?;
        self.readers
            .try_get_with(
                user.clone(),
                self.provision(client, scope, &user, &password),
            )
            .await
            .map_err(QueryAccessError::Cached)
    }

    async fn provision(
        &self,
        client: &Client,
        scope: &QueryScope,
        user: &str,
        password: &str,
    ) -> Result<Connection, QueryAccessError> {
        let database = &self.database;
        if database.is_empty()
            || !database
                .bytes()
                .all(|c| c.is_ascii_alphanumeric() || c == b'_')
        {
            return Err(QueryAccessError::InvalidScope);
        }
        let password_hash = format!("{:x}", Sha256::digest(password));
        self.execute(
            client,
            format!(
                "CREATE USER IF NOT EXISTS {user} IDENTIFIED WITH sha256_hash BY '{password_hash}' \
             SETTINGS readonly = 1 CONST, max_execution_time = 10 CONST, \
             max_result_rows = 1000 CONST, max_result_bytes = 4194304 CONST, \
             result_overflow_mode = 'throw' CONST, max_memory_usage = 268435456 CONST, \
             max_threads = 2 CONST, max_concurrent_queries_for_user = 8 CONST"
            ),
        )
        .await?;
        self.execute(
            client,
            format!("ALTER USER {user} IDENTIFIED WITH sha256_hash BY '{password_hash}'"),
        )
        .await?;
        for table in TABLES {
            let predicate = scope.predicate(table);
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
        for table in TABLES {
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
        .map_err(QueryAccessError::Storage)
    }

    async fn execute(&self, client: &Client, sql: String) -> Result<(), QueryAccessError> {
        let response = client
            .post(self.writer.url().clone())
            .timeout(Duration::from_secs(15))
            .body(sql)
            .send()
            .await
            .map_err(|_| QueryAccessError::ProvisionTransport)?;
        if !response.status().is_success() {
            return Err(QueryAccessError::ProvisionFailed(
                response.status().as_u16(),
            ));
        }
        Ok(())
    }
}

fn credential(secret: &str, purpose: &[u8], identity: &[u8]) -> Result<String, QueryAccessError> {
    let mut mac = Hmac::<Sha256>::new_from_slice(secret.as_bytes())
        .map_err(|_| QueryAccessError::MissingSecret)?;
    mac.update(purpose);
    mac.update(identity);
    Ok(format!("{:x}", mac.finalize().into_bytes()))
}

fn literal(value: &str) -> String {
    format!("'{}'", value.replace('\\', "\\\\").replace('\'', "\\'"))
}
