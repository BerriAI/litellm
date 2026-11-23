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

/// The pools, and what a repository runs its queries on them with.
pub(crate) async fn connect(
    config: &Config,
    secrets: &SecretResolver,
) -> Result<Option<(Pools, Database)>, Error> {
    let database = match litellm_db_resolve::resolve(&config.general_settings, secrets).await {
        Ok(Some(database)) => database,
        Ok(None) => return Ok(None),
        Err(error @ litellm_db_resolve::Error::TokenAuth { .. }) => {
            tracing::warn!("{error}, so the gateway runs without a database pool");
            return Ok(None);
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
    Ok(Some((pools, repositories)))
}
