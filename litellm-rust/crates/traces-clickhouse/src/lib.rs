macro_rules_attribute::attribute_alias! {
    #[apply(wire_type)] =
        #[derive(serde::Serialize, serde::Deserialize)]
        #[cfg_attr(feature = "schema", derive(schemars::JsonSchema))];
    #[apply(response_type)] =
        #[derive(serde::Serialize)]
        #[cfg_attr(feature = "schema", derive(schemars::JsonSchema))];
    #[apply(request_type)] =
        #[derive(serde::Deserialize)]
        #[cfg_attr(feature = "schema", derive(schemars::JsonSchema))];
}

mod config;
mod cost_rows;
mod error;
mod insert;
pub mod query;
mod query_access;
mod reads;
mod receipt;
mod schema;
mod span_batches;
mod span_row;
mod sql;
mod table;
#[cfg(feature = "schema")]
pub mod wire_schema;

pub use config::Config;
pub use error::Error;
pub use insert::project_spend_rows;
pub use insert::{InsertRow, InsertTable, encode_rows, insert_rows, insert_shared_rows};
pub use litellm_storage_clickhouse::{Connection, Parameter};
pub use litellm_traces::{QueryScope, ReadQuery};
pub use query::{QueryHelp, execute_read, query_help, query_sql};
pub use query_access::QueryReaders;
pub use reads::ClickHouseTraces;
pub use receipt::trace_received;
pub use schema::{
    NORMALIZED_FIELD_DEFINITIONS, NormalizedFieldDefinition, apply_migrations, ensure_schema,
    reconcile_retention, schema_statements,
};
pub use span_row::span_rows;
pub use sql::execute_named_read;
pub use table::TraceTable;
