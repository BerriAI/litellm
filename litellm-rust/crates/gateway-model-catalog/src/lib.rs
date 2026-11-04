mod catalog;
mod error;
mod routes;
mod settings;

pub use catalog::{Installed, ModelCatalog};
pub use error::{Error, Unavailable};
pub use routes::router;
pub use settings::Settings;

#[cfg(test)]
mod test_support;
