use std::{error::Error as StdError, str::FromStr};

use litellm_secret_value::SecretValue;
use rstest::fixture;
use sqlx::{PgPool, postgres::PgConnectOptions};

use crate::{Endpoint, PoolLimits, SslMode};

#[fixture]
pub async fn database_pool() -> Result<PgPool, Box<dyn StdError>> {
    let database_url = std::env::var("DATABASE_URL")?;
    Ok(PgPool::connect(&database_url).await?)
}

pub fn database_endpoint() -> Result<Endpoint, Box<dyn StdError>> {
    let database_url = std::env::var("DATABASE_URL")?;
    let options = PgConnectOptions::from_str(&database_url)?;
    Ok(Endpoint {
        host: options.get_host().to_owned(),
        port: options.get_port(),
        username: options.get_username().to_owned(),
        password: url::Url::parse(&database_url)?
            .password()
            .map(SecretValue::new),
        database: options.get_database().unwrap_or("postgres").to_owned(),
        ssl_mode: SslMode::Prefer,
        ssl_root_cert: None,
        options: Vec::new(),
        statement_cache: true,
        pool: PoolLimits::default(),
    })
}
