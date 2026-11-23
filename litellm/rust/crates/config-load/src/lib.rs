mod environ;
mod error;
mod includes;
mod variables;

use std::path::Path;

use litellm_config::Config;
use litellm_secrets::SecretResolver;

pub use error::Error;
pub use variables::{ConfigEnvironment, environment_values};

/// The config at `path` with every file its `include` names merged in and every `os.environ/`
/// reference read through `secrets`, with its `environment_variables` in front of the
/// environment `secrets` reads. That same environment is put under the config, so a key the
/// config leaves unset is read from its variable.
pub async fn load(path: impl AsRef<Path>, secrets: SecretResolver) -> Result<Config, Error> {
    let document = includes::load(path.as_ref())?;
    let process = secrets.environment();
    let config = Config::from_value(environ::resolve(document, secrets).await?)?;
    let environment =
        ConfigEnvironment::new(environment_values(&config.environment_variables)?, process);
    Ok(config.with_environment(&environment))
}
