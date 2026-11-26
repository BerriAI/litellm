mod document;
mod environ;
mod error;
mod includes;
mod variables;

use std::path::Path;

use litellm_config::Config;
use litellm_secrets::SecretResolver;

pub use document::Document;
pub use error::Error;
pub use variables::{ConfigEnvironment, environment_values};

/// The config at `path` alone, as [`Document::resolve`] reads it.
pub async fn load(path: impl AsRef<Path>, secrets: SecretResolver) -> Result<Config, Error> {
    Document::read(path.as_ref())?.resolve(secrets).await
}
