use crate::{Object, Spelled};

#[macro_rules_attribute::apply(crate::config_section)]
pub struct DatabaseSettings {
    pub database_connection_pool_timeout: Option<Spelled<f64>>,
    pub database_extra_connection_params: Option<Object>,
    pub database_disable_prepared_statements: Option<Spelled<bool>>,
    pub database_statement_timeout: Option<Spelled<f64>>,
    pub database_lock_timeout: Option<Spelled<f64>>,
    pub database_idle_in_transaction_session_timeout: Option<Spelled<f64>>,
    pub disable_prisma_schema_update: Option<Spelled<bool>>,
    pub allow_requests_on_db_unavailable: Option<Spelled<bool>>,
    pub supported_db_objects: Option<Box<[String]>>,
    pub use_redis_transaction_buffer: Option<Spelled<bool>>,
}
