use std::{
    sync::{Arc, Mutex, PoisonError},
    time::{Duration, Instant},
};

use litellm_circuit_breaker::{Breaker, Config, Failure};
use sqlx::{PgPool, Postgres, pool::PoolConnection};

use crate::{Endpoint, Error};

const RECOVERY: Duration = Duration::from_secs(30);

type Connection = PoolConnection<Postgres>;

#[derive(Default)]
struct Recycled(Mutex<Option<Instant>>);

impl Recycled {
    fn mark(&self, now: Instant) {
        *self.0.lock().unwrap_or_else(PoisonError::into_inner) = Some(now);
    }

    fn predates(&self, now: Instant, age: Duration) -> bool {
        let Some(recycled_at) = *self.0.lock().unwrap_or_else(PoisonError::into_inner) else {
            return false;
        };
        now.checked_sub(age)
            .is_none_or(|opened_at| opened_at <= recycled_at)
    }
}

#[derive(Clone)]
struct Side {
    pool: PgPool,
    breaker: Breaker,
    recycled: Arc<Recycled>,
}

impl Side {
    fn new(endpoint: &Endpoint) -> Result<Self, Error> {
        let recycled = Arc::new(Recycled::default());
        let pool = endpoint
            .pool_options()?
            .before_acquire({
                let recycled = Arc::clone(&recycled);
                move |_, metadata| {
                    let fresh = !recycled.predates(Instant::now(), metadata.age);
                    Box::pin(async move { Ok(fresh) })
                }
            })
            .connect_lazy_with(endpoint.connect_options());
        Ok(Self {
            pool,
            recycled,
            breaker: Breaker::new(
                Config {
                    failure_threshold: 1,
                    recovery_timeout: RECOVERY,
                    ..Config::default()
                },
                Box::new(|_| {}),
            ),
        })
    }

    async fn acquire_unless_skipped(&self) -> Option<Connection> {
        let admission = self.breaker.admit(Instant::now()).ok()?;
        match self.pool.acquire().await {
            Ok(connection) => {
                admission.succeeded();
                Some(connection)
            }
            Err(_) => {
                admission.failed(Failure::Hard, Instant::now());
                None
            }
        }
    }

    async fn acquire(&self) -> Result<Connection, Error> {
        let result = self.pool.acquire().await;
        if result.is_err() {
            self.breaker.failed(Failure::Hard, Instant::now());
        }
        Ok(result?)
    }
}

async fn acquire_preferring(
    preferred: &Side,
    fallback: Option<&Side>,
) -> Result<Connection, Error> {
    let Some(fallback) = fallback else {
        return preferred.acquire().await;
    };
    match preferred.acquire_unless_skipped().await {
        Some(connection) => Ok(connection),
        None => fallback.acquire().await,
    }
}

#[derive(Clone)]
pub struct Pools {
    writer: Side,
    reader: Option<Side>,
}

impl Pools {
    pub fn new(writer: &Endpoint, reader: Option<&Endpoint>) -> Result<Self, Error> {
        Ok(Self {
            writer: Side::new(writer)?,
            reader: reader.map(Side::new).transpose()?,
        })
    }

    pub fn writer(&self) -> &PgPool {
        &self.writer.pool
    }

    pub fn reader_pool(&self) -> Option<&PgPool> {
        self.reader.as_ref().map(|reader| &reader.pool)
    }

    pub async fn read(&self) -> Result<Connection, Error> {
        match &self.reader {
            Some(reader) => acquire_preferring(reader, Some(&self.writer)).await,
            None => self.writer.acquire().await,
        }
    }

    pub async fn read_after_write(&self) -> Result<Connection, Error> {
        acquire_preferring(&self.writer, self.reader.as_ref()).await
    }

    pub fn recycle_after(&self, error: &Error) -> bool {
        if !error.needs_fresh_connections() {
            return false;
        }
        let now = Instant::now();
        self.writer.recycled.mark(now);
        if let Some(reader) = &self.reader {
            reader.recycled.mark(now);
        }
        true
    }

    pub async fn close(&self) {
        self.writer.pool.close().await;
        if let Some(reader) = &self.reader {
            reader.pool.close().await;
        }
    }
}

#[cfg(test)]
mod tests {
    use std::{
        error::Error as StdError,
        time::{Duration, Instant},
    };

    use rstest::rstest;
    use sqlx::{Connection as _, PgConnection};

    use super::{Pools, Recycled};
    use crate::{Endpoint, Error, ErrorKind, PoolLimits, test_support::database_endpoint};

    #[rstest]
    #[case::never_recycled(None, 5, false)]
    #[case::opened_before_the_recycle(Some(10), 30, true)]
    #[case::opened_after_the_recycle(Some(10), 5, false)]
    fn discards_connections_opened_before_a_recycle(
        #[case] recycled_seconds_ago: Option<u64>,
        #[case] age_seconds: u64,
        #[case] stale: bool,
    ) {
        let now = Instant::now() + Duration::from_secs(60);
        let recycled = Recycled::default();
        if let Some(ago) = recycled_seconds_ago {
            recycled.mark(now - Duration::from_secs(ago));
        }

        assert_eq!(
            recycled.predates(now, Duration::from_secs(age_seconds)),
            stale
        );
    }

    async fn backend_pid(pools: &Pools) -> Result<i32, sqlx::Error> {
        #[expect(clippy::disallowed_methods, reason = "A probe query needs no schema")]
        sqlx::query_scalar("SELECT pg_backend_pid()")
            .fetch_one(&mut *pools.writer().acquire().await?)
            .await
    }

    async fn raised(pools: &Pools, code: &str) -> Result<Error, sqlx::Error> {
        #[expect(
            clippy::disallowed_methods,
            reason = "Test query must raise the requested SQLSTATE"
        )]
        let error = sqlx::query(sqlx::AssertSqlSafe(format!(
            "DO $$ BEGIN RAISE EXCEPTION USING ERRCODE = '{code}'; END $$"
        )))
        .execute(pools.writer())
        .await
        .expect_err("PostgreSQL must raise the requested error");
        Ok(Error::from(error))
    }

    #[rstest]
    #[case::read_only_writer("25006", true)]
    #[case::lock_timeout("55P03", false)]
    #[tokio::test]
    #[ignore = "requires DATABASE_URL pointing to a Prisma-migrated PostgreSQL database"]
    async fn recycles_pooled_connections_only_after_errors_that_poison_them(
        #[case] code: &str,
        #[case] recycled: bool,
    ) -> Result<(), Box<dyn StdError>> {
        let pools = Pools::new(&database_endpoint()?, None)?;
        let before = backend_pid(&pools).await?;

        assert_eq!(pools.recycle_after(&raised(&pools, code).await?), recycled);

        assert_eq!(backend_pid(&pools).await? != before, recycled);
        Ok(())
    }

    #[tokio::test]
    #[ignore = "requires DATABASE_URL pointing to a Prisma-migrated PostgreSQL database"]
    async fn a_schema_change_under_a_cached_plan_is_recovered_by_a_recycle()
    -> Result<(), Box<dyn StdError>> {
        let endpoint = database_endpoint()?;
        let pools = Pools::new(
            &Endpoint {
                pool: PoolLimits {
                    max_connections: 1,
                    ..endpoint.pool
                },
                ..endpoint.clone()
            },
            None,
        )?;
        let table = format!("native_stale_plan_probe_{}", std::process::id());
        let select = format!("SELECT * FROM {table}");
        #[expect(clippy::disallowed_methods, reason = "A scratch table needs no schema")]
        let run = |sql: &str| sqlx::query(sqlx::AssertSqlSafe(sql.to_owned()));
        let mut admin = PgConnection::connect_with(&endpoint.connect_options()).await?;
        run(&format!("CREATE TABLE {table} (id int)"))
            .execute(&mut admin)
            .await?;

        let outcome = async {
            run(&select).fetch_all(pools.writer()).await?;
            run(&format!("ALTER TABLE {table} ADD COLUMN extra int"))
                .execute(&mut admin)
                .await?;
            let stale = Error::from(
                run(&select)
                    .fetch_all(pools.writer())
                    .await
                    .expect_err("the cached plan must be stale"),
            );
            let recycled = pools.recycle_after(&stale);
            run(&select).fetch_all(pools.writer()).await?;
            Ok::<_, Box<dyn StdError>>((stale.kind(), recycled))
        }
        .await;
        run(&format!("DROP TABLE {table}"))
            .execute(&mut admin)
            .await?;

        assert_eq!(outcome?, (ErrorKind::StalePlan, true));
        Ok(())
    }

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
    async fn reads_go_to_the_reader_and_read_after_write_to_the_writer()
    -> Result<(), Box<dyn StdError>> {
        let writer = database_endpoint()?;
        let pools = Pools::new(&writer, Some(&read_only(&writer)))?;

        assert!(is_read_only(&mut *pools.read().await?).await?);
        assert!(!is_read_only(&mut *pools.read_after_write().await?).await?);
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
    async fn every_read_uses_the_writer_without_a_reader() -> Result<(), Box<dyn StdError>> {
        let pools = Pools::new(&database_endpoint()?, None)?;

        assert!(pools.reader_pool().is_none());
        assert!(!is_read_only(&mut *pools.read().await?).await?);
        assert!(!is_read_only(&mut *pools.read_after_write().await?).await?);
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

    #[tokio::test]
    #[ignore = "requires DATABASE_URL pointing to a Prisma-migrated PostgreSQL database"]
    async fn reads_stay_on_the_reader_while_the_writer_is_down() -> Result<(), Box<dyn StdError>> {
        let reader = read_only(&database_endpoint()?);
        let pools = Pools::new(&unreachable(&reader), Some(&reader))?;

        assert!(is_read_only(&mut *pools.read().await?).await?);
        assert!(pools.writer().acquire().await.is_err());
        Ok(())
    }

    #[tokio::test]
    #[ignore = "requires DATABASE_URL pointing to a Prisma-migrated PostgreSQL database"]
    async fn read_after_write_yields_to_the_reader_while_the_writer_is_down()
    -> Result<(), Box<dyn StdError>> {
        let reader = read_only(&database_endpoint()?);
        let pools = Pools::new(&unreachable(&reader), Some(&reader))?;

        assert!(is_read_only(&mut *pools.read_after_write().await?).await?);
        let started = Instant::now();
        assert!(is_read_only(&mut *pools.read_after_write().await?).await?);

        assert!(started.elapsed().as_secs_f64() < 1.0);
        Ok(())
    }

    #[tokio::test]
    #[ignore = "requires DATABASE_URL pointing to a Prisma-migrated PostgreSQL database"]
    async fn read_after_write_fails_when_the_only_database_is_down() -> Result<(), Box<dyn StdError>>
    {
        let pools = Pools::new(&unreachable(&database_endpoint()?), None)?;

        assert!(pools.read_after_write().await.is_err());
        Ok(())
    }
}
