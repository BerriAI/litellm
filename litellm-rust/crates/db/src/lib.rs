mod error;
mod queries;
mod schema;

pub use error::Error;
pub use schema::{REQUIRED_MIGRATION, SchemaCompatibility, schema_compatibility};
