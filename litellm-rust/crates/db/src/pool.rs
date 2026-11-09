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
    writer_persistent: bool,
    reader_persistent: bool,
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
            writer_persistent: writer.statement_cache_capacity > 0,
            reader_persistent: reader
                .as_ref()
                .is_some_and(|settings| settings.statement_cache_capacity > 0),
        })
    }

    pub fn reader_available(&self) -> bool {
        self.reader_available
    }

    pub fn persistent(&self, writer: bool) -> bool {
        if !writer && self.reader_available {
            self.reader_persistent
        } else {
            self.writer_persistent
        }
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

#[cfg(test)]
mod tests {
    use std::{error::Error as StdError, time::Duration};

    use litellm_secret_value::SecretValue;
    use rstest::{fixture, rstest};
    use sqlx::{ConnectOptions, PgPool, postgres::PgPoolOptions};

    use super::{DatabasePools, DatabaseSettings, acquire};
    use crate::test_support::database_pool;

    #[fixture]
    async fn settings() -> Result<DatabaseSettings, Box<dyn StdError>> {
        let pool = database_pool().await?;
        let options = pool.connect_options();
        let settings = DatabaseSettings {
            host: options.get_socket().map_or_else(
                || options.get_host().to_owned(),
                |socket| socket.to_string_lossy().into_owned(),
            ),
            port: options.get_port(),
            username: options.get_username().into(),
            password: options.to_url_lossy().password().map(SecretValue::from),
            database: options
                .get_database()
                .ok_or("database must be explicit")?
                .into(),
            schema: "public".into(),
            ssl_mode: options.get_ssl_mode(),
            ssl_root_cert: None,
            max_connections: 1,
            acquire_timeout: Duration::from_secs(2),
            connect_timeout: Duration::from_secs(2),
            idle_timeout: None,
            statement_cache_capacity: 0,
        };
        pool.close().await;
        Ok(settings)
    }

    #[rstest]
    #[case::writer_only(false, false)]
    #[case::reader_healthy(true, false)]
    #[case::reader_unavailable(true, true)]
    #[tokio::test]
    #[ignore = "requires DATABASE_URL pointing to a Prisma-migrated PostgreSQL database"]
    async fn reader_selection_and_shutdown(
        #[case] configured: bool,
        #[case] unavailable: bool,
        #[future(awt)] settings: Result<DatabaseSettings, Box<dyn StdError>>,
    ) -> Result<(), Box<dyn StdError>> {
        let settings = DatabaseSettings {
            statement_cache_capacity: 1,
            ..settings?
        };
        let reader = configured.then(|| DatabaseSettings {
            database: if unavailable {
                "native_auth_missing_database".into()
            } else {
                settings.database.clone()
            },
            connect_timeout: Duration::from_millis(100),
            statement_cache_capacity: 0,
            ..settings.clone()
        });
        let pools = DatabasePools::connect(settings, reader).await?;
        assert_eq!(pools.reader_available(), configured && !unavailable);
        assert!(pools.persistent(true));
        assert_eq!(pools.persistent(false), !pools.reader_available());
        let mut writer = pools.acquire(true).await?;
        #[expect(
            clippy::disallowed_methods,
            reason = "Test verifies connection provenance rather than a production query"
        )]
        let (writer_pid, app, schema): (i32, String, String) = sqlx::query_as(
            "SELECT pg_backend_pid(), current_setting('application_name'), current_schema()",
        )
        .fetch_one(&mut *writer)
        .await?;
        assert_eq!(app, "native-auth");
        assert_eq!(schema, "public");
        drop(writer);
        let mut reader = pools.acquire(false).await?;
        #[expect(
            clippy::disallowed_methods,
            reason = "Test verifies pool selection by backend identity"
        )]
        let reader_pid: i32 = sqlx::query_scalar("SELECT pg_backend_pid()")
            .fetch_one(&mut *reader)
            .await?;
        assert_eq!(writer_pid == reader_pid, !configured || unavailable);
        drop(reader);
        let clone = pools.clone();
        pools.close().await;
        clone.close().await;
        assert!(matches!(
            clone.acquire(false).await,
            Err(crate::Error::Query(sqlx::Error::PoolClosed))
        ));
        assert!(
            clone
                .set_passwords(Some(SecretValue::from("next")), None)
                .is_err()
        );
        Ok(())
    }

    #[rstest]
    #[case::uncached(0, 0)]
    #[case::cached(4, 2)]
    #[tokio::test]
    #[ignore = "requires DATABASE_URL pointing to a Prisma-migrated PostgreSQL database"]
    async fn respects_statement_persistence(
        #[case] capacity: usize,
        #[case] expected: i64,
        #[future(awt)] settings: Result<DatabaseSettings, Box<dyn StdError>>,
    ) -> Result<(), Box<dyn StdError>> {
        let settings = DatabaseSettings {
            statement_cache_capacity: capacity,
            ..settings?
        };
        let pools = DatabasePools::connect(settings, None).await?;
        let mut connection = pools.acquire(false).await?;
        crate::auth_read(
            &mut *connection,
            crate::AuthRead::User,
            &[],
            pools.persistent(false),
        )
        .await?;
        crate::auth_spend(
            &mut *connection,
            crate::AuthSpend::User,
            "missing",
            pools.persistent(false),
        )
        .await?;
        #[expect(
            clippy::disallowed_methods,
            reason = "Test observes server-side prepared statements on the selected connection"
        )]
        let prepared: i64 = sqlx::query_scalar("SELECT count(*) FROM pg_prepared_statements")
            .persistent(false)
            .fetch_one(&mut *connection)
            .await?;
        assert_eq!(prepared, expected);
        drop(connection);
        pools.close().await;
        Ok(())
    }

    #[rstest]
    #[case::reader_survives(true)]
    #[case::both_unavailable(false)]
    #[tokio::test]
    #[ignore = "requires DATABASE_URL pointing to a Prisma-migrated PostgreSQL database"]
    async fn unavailable_writer(
        #[case] healthy_reader: bool,
        #[future(awt)] settings: Result<DatabaseSettings, Box<dyn StdError>>,
    ) -> Result<(), Box<dyn StdError>> {
        let settings = settings?;
        let writer = DatabaseSettings {
            database: "native_auth_missing_database".into(),
            connect_timeout: Duration::from_millis(100),
            acquire_timeout: Duration::from_millis(100),
            ..settings.clone()
        };
        let result = DatabasePools::connect(writer, healthy_reader.then_some(settings)).await;
        if !healthy_reader {
            assert!(result.is_err());
            return Ok(());
        }
        let pools = result?;
        assert!(pools.reader_available());
        assert!(pools.acquire(false).await.is_ok());
        assert!(pools.acquire(true).await.is_err());
        pools.close().await;
        Ok(())
    }

    #[rstest]
    #[case::bounded(false)]
    #[case::unbounded(true)]
    #[tokio::test]
    #[ignore = "requires DATABASE_URL pointing to a Prisma-migrated PostgreSQL database"]
    async fn acquisition_deadline_and_cancellation(
        #[case] unbounded: bool,
        #[future(awt)] database_pool: Result<PgPool, Box<dyn StdError>>,
    ) -> Result<(), Box<dyn StdError>> {
        let source = database_pool?;
        let pool = PgPoolOptions::new()
            .max_connections(1)
            .acquire_timeout(Duration::from_millis(20))
            .connect_with((*source.connect_options()).clone())
            .await?;
        let held = pool.acquire().await?;
        let result =
            tokio::time::timeout(Duration::from_millis(150), acquire(&pool, unbounded)).await;
        if unbounded {
            assert!(result.is_err());
        } else {
            assert!(matches!(result, Ok(Err(sqlx::Error::PoolTimedOut))));
        }
        drop(held);
        let recovered =
            tokio::time::timeout(Duration::from_secs(2), acquire(&pool, unbounded)).await??;
        drop(recovered);
        pool.close().await;
        assert!(matches!(
            acquire(&pool, unbounded).await,
            Err(sqlx::Error::PoolClosed)
        ));
        Ok(())
    }

    #[rstest]
    #[tokio::test]
    #[ignore = "requires DATABASE_URL pointing to a Prisma-migrated PostgreSQL database"]
    async fn query_error_does_not_replay_on_writer(
        #[future(awt)] settings: Result<DatabaseSettings, Box<dyn StdError>>,
    ) -> Result<(), Box<dyn StdError>> {
        let settings = settings?;
        let reader = DatabaseSettings {
            schema: "native_auth_missing_schema".into(),
            ..settings.clone()
        };
        let pools = DatabasePools::connect(settings, Some(reader)).await?;
        let mut read = pools.acquire(false).await?;
        assert!(
            crate::auth_read(
                &mut *read,
                crate::AuthRead::User,
                &[],
                pools.persistent(false)
            )
            .await
            .is_err()
        );
        drop(read);
        let mut write = pools.acquire(true).await?;
        assert!(
            crate::auth_read(
                &mut *write,
                crate::AuthRead::User,
                &[],
                pools.persistent(true)
            )
            .await?
            .is_empty()
        );
        drop(write);
        pools.close().await;
        Ok(())
    }

    #[rstest]
    #[tokio::test]
    #[ignore = "requires DATABASE_URL pointing to a Prisma-migrated PostgreSQL database"]
    async fn zero_timeouts_and_secret_redaction(
        #[future(awt)] settings: Result<DatabaseSettings, Box<dyn StdError>>,
    ) -> Result<(), Box<dyn StdError>> {
        let settings = DatabaseSettings {
            acquire_timeout: Duration::ZERO,
            connect_timeout: Duration::ZERO,
            ..settings?
        };
        let debug = format!(
            "{:?}",
            DatabaseSettings {
                password: Some(SecretValue::from("sensitive-test-password")),
                ..settings.clone()
            }
        );
        assert!(!debug.contains("sensitive-test-password"));
        assert!(debug.contains("[REDACTED]"));
        let pools = DatabasePools::connect(settings, None).await?;
        assert!(!pools.reader_available());
        assert!(pools.acquire(false).await.is_ok());
        assert!(
            pools
                .set_passwords(None, Some(SecretValue::from("reader")))
                .is_err()
        );
        pools.close().await;
        Ok(())
    }
}
