use std::sync::Arc;

use litellm_config::Config;
use litellm_db::{Observer, Outcome, Pools, QueryEvent};
use litellm_repositories::Database;
use litellm_secrets::SecretResolver;

use crate::Error;

struct Logged;

impl Observer for Logged {
    fn record(&self, event: QueryEvent) {
        let failure = match &event.outcome {
            Outcome::Succeeded => None,
            Outcome::Failed(failure) => Some(failure.as_str()),
        };
        tracing::debug!(
            call = event.call.call_type,
            table = event.call.table,
            duration = ?event.duration,
            failure,
            "database operation"
        );
    }
}

/// What the gateway reads the proxy's database through.
pub(crate) enum Connected {
    /// The pools, and what a repository runs its queries on them with.
    Pools(Pools, Database),
    /// The config names no database, so the proxy reads none either.
    NoDatabase,
    /// The config names a database the gateway cannot sign in to (token auth), so the proxy holds
    /// rows the gateway cannot read.
    Unread,
}

impl Connected {
    pub(crate) fn pools(self) -> Option<(Pools, Database)> {
        match self {
            Self::Pools(pools, repositories) => Some((pools, repositories)),
            Self::NoDatabase | Self::Unread => None,
        }
    }
}

pub(crate) async fn connect(config: &Config, secrets: &SecretResolver) -> Result<Connected, Error> {
    let database = match litellm_db_resolve::resolve(&config.general_settings, secrets).await {
        Ok(Some(database)) => database,
        Ok(None) => return Ok(Connected::NoDatabase),
        Err(error @ litellm_db_resolve::Error::TokenAuth { .. }) => {
            tracing::warn!("{error}, so the gateway runs without a database pool");
            return Ok(Connected::Unread);
        }
        Err(error) => return Err(error.into()),
    };
    for ignored in &database.ignored {
        tracing::warn!(
            "Database setting {} is ignored by the native pool: {}",
            ignored.setting,
            ignored.consequence
        );
    }
    let pools = Pools::new(&database.writer, database.reader.as_ref())?;
    let names_statements = database.writer.names_statements()
        && database
            .reader
            .as_ref()
            .is_none_or(litellm_db::Endpoint::names_statements);
    let repositories = Database::new(
        pools.clone(),
        Arc::new(Logged),
        database.writer.socket_timeout()?,
        names_statements,
    );
    Ok(Connected::Pools(pools, repositories))
}
