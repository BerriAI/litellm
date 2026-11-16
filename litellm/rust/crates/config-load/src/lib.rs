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
/// environment `secrets` reads.
pub async fn load(path: impl AsRef<Path>, secrets: SecretResolver) -> Result<Config, Error> {
    let document = includes::load(path.as_ref())?;
    Ok(Config::from_value(
        environ::resolve(document, secrets).await?,
    )?)
}
