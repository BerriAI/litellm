use std::time::Duration;

use litellm_secret_value::SecretValue;
use sqlx::{
    ConnectOptions, PgPool,
    postgres::{PgConnectOptions, PgPoolOptions, PgSslMode},
};

use crate::Error;

#[derive(Clone, Debug)]
pub struct DatabaseSettings {
    pub host: String,
    pub port: u16,
    pub username: String,
    pub password: Option<SecretValue>,
    pub database: String,
    pub schema: String,
    pub ssl_mode: PgSslMode,
    pub ssl_root_cert: Option<String>,
    pub max_connections: u32,
    pub acquire_timeout: Duration,
    pub connect_timeout: Duration,
    pub idle_timeout: Option<Duration>,
    pub statement_cache_capacity: usize,
}

impl DatabaseSettings {
    fn pool(&self) -> Result<PgPool, Error> {
        if self.max_connections == 0 {
            return Err(Error::InvalidSetting("max_connections"));
        }
        if self.host.is_empty()
            || self.username.is_empty()
            || self.database.is_empty()
            || self.schema.is_empty()
        {
            return Err(Error::InvalidSetting("connection identity"));
        }
        let options = PgConnectOptions::new_without_pgpass()
            .host(&self.host)
            .port(self.port)
            .username(&self.username)
            .password(self.password.as_ref().map_or("", SecretValue::expose))
            .database(&self.database)
            .application_name("native-auth")
            .ssl_mode(self.ssl_mode)
            .statement_cache_capacity(self.statement_cache_capacity)
            .options([(
                "search_path",
                format!("\"{}\"", self.schema.replace('"', "\"\"")),
            )])
            .disable_statement_logging();
        let options = match &self.ssl_root_cert {
            Some(cert) => options.ssl_root_cert(cert),
            None => options,
        };
        Ok(PgPoolOptions::new()
            .max_connections(self.max_connections)
            .acquire_timeout(if self.acquire_timeout.is_zero() {
                Duration::from_secs(30)
            } else {
                self.acquire_timeout
            })
            .idle_timeout(self.idle_timeout)
            .connect_lazy_with(options))
    }
}

#[derive(Clone, Debug)]
pub struct DatabasePools {
    writer: PgPool,
    reader: Option<PgPool>,
    reader_available: bool,
    writer_unbounded: bool,
    reader_unbounded: bool,
}

async fn acquire(
    pool: &PgPool,
    unbounded: bool,
) -> Result<sqlx::pool::PoolConnection<sqlx::Postgres>, sqlx::Error> {
    loop {
        match pool.acquire().await {
            Err(sqlx::Error::PoolTimedOut) if unbounded => continue,
            result => return result,
        }
    }
}

async fn connect_pool(pool: &PgPool, settings: &DatabaseSettings) -> Result<(), Error> {
    let connection = acquire(pool, settings.acquire_timeout.is_zero());
    if settings.connect_timeout.is_zero() {
        connection.await?;
    } else {
        tokio::time::timeout(settings.connect_timeout, connection).await??;
    }
    Ok(())
}

impl DatabasePools {
    pub async fn connect(
        writer: DatabaseSettings,
        reader: Option<DatabaseSettings>,
    ) -> Result<Self, Error> {
        let writer_pool = writer.pool()?;
        let reader_pool = reader.as_ref().map(DatabaseSettings::pool).transpose()?;
        let writer_result = connect_pool(&writer_pool, &writer).await;
        let reader_result = match (&reader_pool, &reader) {
            (Some(pool), Some(settings)) => Some(connect_pool(pool, settings).await),
            _ => None,
        };
        let reader_available = reader_result.as_ref().is_some_and(Result::is_ok);
        drop(reader_result);
        if let Err(error) = writer_result
            && !reader_available
        {
            writer_pool.close().await;
            if let Some(pool) = reader_pool {
                pool.close().await;
            }
            return Err(error);
        }
        Ok(Self {
            writer: writer_pool,
            reader: reader_pool,
            reader_available,
            writer_unbounded: writer.acquire_timeout.is_zero(),
            reader_unbounded: reader
                .as_ref()
                .is_some_and(|settings| settings.acquire_timeout.is_zero()),
        })
    }

    pub fn reader_available(&self) -> bool {
        self.reader_available
    }

    pub fn pool(&self, writer: bool) -> &PgPool {
        match (&self.reader, writer, self.reader_available) {
            (Some(pool), false, true) => pool,
            _ => &self.writer,
        }
    }

    pub async fn acquire(
        &self,
        writer: bool,
    ) -> Result<sqlx::pool::PoolConnection<sqlx::Postgres>, Error> {
        let unbounded = if !writer && self.reader_available {
            self.reader_unbounded
        } else {
            self.writer_unbounded
        };
        Ok(acquire(self.pool(writer), unbounded).await?)
    }

    pub fn set_passwords(
        &self,
        writer: Option<SecretValue>,
        reader: Option<SecretValue>,
    ) -> Result<(), Error> {
        if self.writer.is_closed() {
            return Err(sqlx::Error::PoolClosed.into());
        }
        if reader.is_some() && self.reader.is_none() {
            return Err(Error::InvalidSetting("reader_password"));
        }
        if let Some(password) = writer {
            self.writer.set_connect_options(
                (*self.writer.connect_options())
                    .clone()
                    .password(password.expose()),
            );
        }
        if let (Some(pool), Some(password)) = (&self.reader, reader) {
            pool.set_connect_options(
                (*pool.connect_options())
                    .clone()
                    .password(password.expose()),
            );
        }
        Ok(())
    }

    pub async fn close(&self) {
        if let Some(reader) = &self.reader {
            tokio::join!(self.writer.close(), reader.close());
        } else {
            self.writer.close().await;
        }
    }
}
