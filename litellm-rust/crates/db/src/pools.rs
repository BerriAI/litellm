use std::time::{Duration, Instant};

use litellm_circuit_breaker::{Breaker, Config, Failure};
use sqlx::{PgPool, Postgres, pool::PoolConnection};

use crate::{Endpoint, Error};

const READER_RECOVERY: Duration = Duration::from_secs(30);

#[derive(Clone)]
struct Reader {
    pool: PgPool,
    breaker: Breaker,
}

#[derive(Clone)]
pub struct Pools {
    writer: PgPool,
    reader: Option<Reader>,
}

impl Pools {
    pub fn new(writer: &Endpoint, reader: Option<&Endpoint>) -> Result<Self, Error> {
        Ok(Self {
            writer: writer.pool()?,
            reader: reader
                .map(|endpoint| {
                    endpoint.pool().map(|pool| Reader {
                        pool,
                        breaker: Breaker::new(
                            Config {
                                failure_threshold: 1,
                                recovery_timeout: READER_RECOVERY,
                                ..Config::default()
                            },
                            Box::new(|_| {}),
                        ),
                    })
                })
                .transpose()?,
        })
    }

    pub fn writer(&self) -> &PgPool {
        &self.writer
    }

    pub fn reader_pool(&self) -> Option<&PgPool> {
        self.reader.as_ref().map(|reader| &reader.pool)
    }

    pub async fn read(&self) -> Result<PoolConnection<Postgres>, Error> {
        let Some(reader) = &self.reader else {
            return Ok(self.writer.acquire().await?);
        };
        let Ok(admission) = reader.breaker.admit(Instant::now()) else {
            return Ok(self.writer.acquire().await?);
        };
        match reader.pool.acquire().await {
            Ok(connection) => {
                admission.succeeded();
                Ok(connection)
            }
            Err(_) => {
                admission.failed(Failure::Hard, Instant::now());
                Ok(self.writer.acquire().await?)
            }
        }
    }

    pub async fn close(&self) {
        self.writer.close().await;
        if let Some(reader) = &self.reader {
            reader.pool.close().await;
        }
    }
}

#[cfg(test)]
mod tests {
    use std::{error::Error as StdError, time::Instant};

    use sqlx::PgConnection;

    use super::Pools;
    use crate::{Endpoint, PoolLimits, test_support::database_endpoint};

    fn read_only(endpoint: &Endpoint) -> Endpoint {
        Endpoint {
            options: vec![("default_transaction_read_only".to_owned(), "on".to_owned())],
            ..endpoint.clone()
        }
    }

    fn unreachable(endpoint: &Endpoint) -> Endpoint {
        Endpoint {
            host: "127.0.0.1".to_owned(),
            port: 1,
            pool: PoolLimits {
                acquire_timeout_seconds: Some(2.0),
                ..endpoint.pool
            },
            ..endpoint.clone()
        }
    }

    async fn is_read_only(connection: &mut PgConnection) -> Result<bool, sqlx::Error> {
        #[expect(
            clippy::disallowed_methods,
            reason = "Session settings are not part of the compile-checked schema"
        )]
        let setting: String = sqlx::query_scalar("SHOW default_transaction_read_only")
            .fetch_one(connection)
            .await?;
        Ok(setting == "on")
    }

    #[tokio::test]
    #[ignore = "requires DATABASE_URL pointing to a Prisma-migrated PostgreSQL database"]
    async fn reads_go_to_the_reader() -> Result<(), Box<dyn StdError>> {
        let writer = database_endpoint()?;
        let pools = Pools::new(&writer, Some(&read_only(&writer)))?;

        assert!(is_read_only(&mut *pools.read().await?).await?);
        assert!(!is_read_only(&mut *pools.writer().acquire().await?).await?);
        Ok(())
    }

    #[tokio::test]
    #[ignore = "requires DATABASE_URL pointing to a Prisma-migrated PostgreSQL database"]
    async fn writes_sent_to_the_reader_fail_read_only() -> Result<(), Box<dyn StdError>> {
        let writer = database_endpoint()?;
        let pools = Pools::new(&writer, Some(&read_only(&writer)))?;
        let mut connection = pools.read().await?;

        #[expect(clippy::disallowed_methods, reason = "A scratch table needs no schema")]
        let error = sqlx::query("CREATE TABLE native_reader_write_probe (id int)")
            .execute(&mut *connection)
            .await
            .expect_err("a read-only session must reject writes");

        assert_eq!(
            error.as_database_error().and_then(|error| error.code()),
            Some("25006".into())
        );
        Ok(())
    }

    #[tokio::test]
    #[ignore = "requires DATABASE_URL pointing to a Prisma-migrated PostgreSQL database"]
    async fn reads_use_the_writer_without_a_reader() -> Result<(), Box<dyn StdError>> {
        let pools = Pools::new(&database_endpoint()?, None)?;

        assert!(pools.reader_pool().is_none());
        assert!(!is_read_only(&mut *pools.read().await?).await?);
        Ok(())
    }

    #[tokio::test]
    #[ignore = "requires DATABASE_URL pointing to a Prisma-migrated PostgreSQL database"]
    async fn unreachable_reader_falls_back_once_then_skips_it() -> Result<(), Box<dyn StdError>> {
        let writer = database_endpoint()?;
        let pools = Pools::new(&writer, Some(&unreachable(&writer)))?;

        assert!(!is_read_only(&mut *pools.read().await?).await?);
        let started = Instant::now();
        assert!(!is_read_only(&mut *pools.read().await?).await?);

        assert!(started.elapsed().as_secs_f64() < 1.0);
        Ok(())
    }
}
