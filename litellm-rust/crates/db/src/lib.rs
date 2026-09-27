#[cfg(feature = "schema")]
pub mod codegen;
mod failure;
mod queries;

pub use failure::DbFailure;
#[cfg(feature = "schema")]
pub use queries::CATALOG;
