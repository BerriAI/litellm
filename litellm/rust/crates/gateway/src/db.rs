use litellm_config::Config;
use litellm_db::Pools;
use litellm_secrets::SecretResolver;

use crate::Error;

pub(crate) async fn connect(
    config: &Config,
    secrets: &SecretResolver,
) -> Result<Option<Pools>, Error> {
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
    Ok(Some(Pools::new(
        &database.writer,
        database.reader.as_ref(),
    )?))
}
