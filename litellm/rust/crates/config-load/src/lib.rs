mod error;
mod includes;

use std::path::Path;

use litellm_config::Config;

pub use error::Error;

/// The config at `path` with every file its `include` names merged in.
pub fn load(path: impl AsRef<Path>) -> Result<Config, Error> {
    Ok(Config::from_value(includes::load(path.as_ref())?)?)
}
