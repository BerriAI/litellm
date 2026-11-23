use std::{
    io,
    sync::{Arc, Mutex, PoisonError},
    time::{Duration, Instant},
};

use litellm_circuit_breaker::{Breaker, Config, Failure};
use litellm_secret_value::SecretValue;
use sqlx::{PgPool, Postgres, pool::PoolConnection};
use tokio::{net::TcpStream, sync::Mutex as AsyncMutex};

use crate::{Endpoint, Error, endpoint::ClientFiles};

const RECOVERY: Duration = Duration::from_secs(30);
/// How long an acquire runs before it checks that the server accepts connections at all. sqlx retries a
/// refused connection until the acquire timeout, Prisma's `pool_timeout` (60s by default), while Prisma
/// fails it at once, so without this check a stopped database stalls every lookup instead of failing it.
const REFUSAL_CHECK_AFTER: Duration = Duration::from_millis(100);
const REFUSAL_PROBE_REUSE: Duration = Duration::from_secs(1);

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

struct RefusalWatch {
    host: String,
    port: u16,
    last: AsyncMutex<Option<(Instant, bool)>>,
}

impl RefusalWatch {
    fn of(endpoint: &Endpoint) -> Option<Self> {
        endpoint.socket.is_none().then(|| Self {
            host: endpoint.host.clone(),
            port: endpoint.port,
            last: AsyncMutex::default(),
        })
    }

    async fn refused(&self) -> io::Error {
        loop {
            tokio::time::sleep(REFUSAL_CHECK_AFTER).await;
            if self.refuses().await {
                return io::ErrorKind::ConnectionRefused.into();
            }
        }
    }

    async fn refuses(&self) -> bool {
        let mut last = self.last.lock().await;
        if let Some((probed_at, refused)) = *last
            && probed_at.elapsed() < REFUSAL_PROBE_REUSE
        {
            return refused;
        }
        let refused = matches!(
            TcpStream::connect((self.host.as_str(), self.port)).await,
            Err(error) if error.kind() == io::ErrorKind::ConnectionRefused
        );
        *last = Some((Instant::now(), refused));
        refused
    }
}

#[derive(Clone)]
struct FailFastPool {
    pool: PgPool,
    watch: Option<Arc<RefusalWatch>>,
}

impl FailFastPool {
    async fn acquire(&self) -> Result<Connection, sqlx::Error> {
        let Some(watch) = &self.watch else {
            return self.pool.acquire().await;
        };
        tokio::select! {
            biased;
            acquired = self.pool.acquire() => acquired,
            refused = watch.refused() => Err(sqlx::Error::Io(refused)),
        }
    }
}

/// The endpoint a pool's connect options are built from, with its current password, and the client certificate files
/// as they were when the options were built.
struct Connect {
    endpoint: Endpoint,
    files: ClientFiles,
}

#[derive(Clone)]
struct Side {
    pool: FailFastPool,
    connect: Arc<Mutex<Connect>>,
    breaker: Breaker,
    recycled: Arc<Recycled>,
}

impl Side {
    fn new(endpoint: &Endpoint) -> Result<Self, Error> {
        let recycled = Arc::new(Recycled::default());
        let files = ClientFiles::of(endpoint);
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
            pool: FailFastPool {
                pool,
                watch: RefusalWatch::of(endpoint).map(Arc::new),
            },
            connect: Arc::new(Mutex::new(Connect {
                endpoint: endpoint.clone(),
                files,
            })),
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
        match self.pool().acquire().await {
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
        let result = self.pool().acquire().await;
        if result.is_err() {
            self.breaker.failed(Failure::Hard, Instant::now());
        }
        Ok(result?)
    }

    fn set_password(&self, password: SecretValue) {
        let mut connect = self.connect.lock().unwrap_or_else(PoisonError::into_inner);
        connect.endpoint = connect.endpoint.with_password(password);
        connect.files = ClientFiles::of(&connect.endpoint);
        self.pool
            .pool
            .set_connect_options(connect.endpoint.connect_options());
    }

    /// The pool, with its connect options built again first when `PGSSLCERT`, `PGSSLKEY` or the root certificate
    /// changed since they were built, so a new connection uses the files as they are now even when the options hold a
    /// key converted for native-tls or a root certificate in place of one native-tls cannot load.
    fn pool(&self) -> &FailFastPool {
        let mut connect = self.connect.lock().unwrap_or_else(PoisonError::into_inner);
        let files = ClientFiles::of(&connect.endpoint);
        if connect.files != files {
            connect.files = files;
            self.pool
                .pool
                .set_connect_options(connect.endpoint.connect_options());
        }
        &self.pool
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
        &self.writer.pool().pool
    }

    pub fn reader_pool(&self) -> Option<&PgPool> {
        self.reader.as_ref().map(|reader| &reader.pool().pool)
    }

    pub fn set_writer_password(&self, password: SecretValue) {
        self.writer.set_password(password);
    }

    /// Whether a read replica is configured to take the password.
    pub fn set_reader_password(&self, password: SecretValue) -> bool {
        self.reader
            .as_ref()
            .map(|reader| reader.set_password(password))
            .is_some()
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

    /// A connection from the writer alone, failing while the writer is down rather than serving a read
    /// replica that may lag behind its latest commit.
    pub async fn writer_only(&self) -> Result<Connection, Error> {
        self.writer.acquire().await
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
        self.writer.pool.pool.close().await;
        if let Some(reader) = &self.reader {
            reader.pool.pool.close().await;
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

    use super::{Pools, Recycled, RefusalWatch};
    use crate::{Endpoint, Error, ErrorKind, PoolLimits, SslMode, test_support::database_endpoint};

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
    async fn writer_only_fails_while_the_writer_is_down_despite_a_healthy_reader()
    -> Result<(), Box<dyn StdError>> {
        let reader = read_only(&database_endpoint()?);
        let pools = Pools::new(&unreachable(&reader), Some(&reader))?;

        assert!(pools.writer_only().await.is_err());
        assert!(pools.writer_only().await.is_err());
        assert!(is_read_only(&mut *pools.read().await?).await?);
        Ok(())
    }

    #[tokio::test]
    #[ignore = "requires DATABASE_URL pointing to a Prisma-migrated PostgreSQL database"]
    async fn writer_only_uses_the_writer_when_a_reader_is_configured()
    -> Result<(), Box<dyn StdError>> {
        let writer = database_endpoint()?;
        let pools = Pools::new(&writer, Some(&read_only(&writer)))?;

        assert!(!is_read_only(&mut *pools.writer_only().await?).await?);
        Ok(())
    }

    #[rstest]
    #[tokio::test]
    async fn concurrent_waiters_share_one_refusal_probe() -> Result<(), Box<dyn StdError>> {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await?;
        let port = listener.local_addr()?.port();
        let accepted = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let counter = std::sync::Arc::clone(&accepted);
        let server = tokio::spawn(async move {
            while listener.accept().await.is_ok() {
                counter.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            }
        });
        let watch = RefusalWatch::of(&local_endpoint(port)).ok_or("a TCP endpoint has a watch")?;

        let verdicts = futures_util::future::join_all((0..20).map(|_| watch.refuses())).await;
        tokio::time::sleep(Duration::from_millis(50)).await;
        server.abort();

        assert!(verdicts.iter().all(|refused| !refused));
        assert_eq!(accepted.load(std::sync::atomic::Ordering::SeqCst), 1);
        Ok(())
    }

    fn local_endpoint(port: u16) -> Endpoint {
        Endpoint {
            host: "127.0.0.1".to_owned(),
            port,
            socket: None,
            username: "litellm".to_owned(),
            password: None,
            database: "proxy".to_owned(),
            ssl_mode: SslMode::Disable,
            ssl_root_cert: None,
            ssl_client: None,
            application_name: None,
            options: Vec::new(),
            statement_cache_capacity: 0,
            pool: PoolLimits {
                max_connections: 1,
                acquire_timeout_seconds: Some(60.0),
                idle_timeout_seconds: None,
                socket_timeout_seconds: None,
            },
        }
    }

    #[rstest]
    #[tokio::test]
    async fn a_waiter_fails_once_the_server_starts_refusing() -> Result<(), Box<dyn StdError>> {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await?;
        let port = listener.local_addr()?.port();
        let watch = RefusalWatch::of(&local_endpoint(port)).ok_or("a TCP endpoint has a watch")?;
        let started = Instant::now();
        let waiter = tokio::spawn(async move { watch.refused().await });

        tokio::time::sleep(Duration::from_millis(300)).await;
        drop(listener);
        let refused = tokio::time::timeout(Duration::from_secs(5), waiter).await??;

        assert_eq!(refused.kind(), std::io::ErrorKind::ConnectionRefused);
        assert!(started.elapsed() > Duration::from_millis(300));
        Ok(())
    }

    #[rstest]
    #[tokio::test]
    async fn a_refusing_server_fails_the_acquire_long_before_its_timeout()
    -> Result<(), Box<dyn StdError>> {
        let closed_port = std::net::TcpListener::bind("127.0.0.1:0")?
            .local_addr()?
            .port();
        let refusing = local_endpoint(closed_port);
        let pools = Pools::new(&refusing, None)?;
        let started = Instant::now();

        let error = pools
            .writer_only()
            .await
            .err()
            .ok_or("a refusing server handed out a connection")?;

        assert_eq!(error.kind(), ErrorKind::Unreachable);
        assert!(started.elapsed() < Duration::from_secs(5));
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

    #[cfg(feature = "native-tls")]
    mod native_tls_client_files {
        use std::{
            error::Error as StdError,
            io::{Read, Write},
            net::TcpListener,
            path::{Path, PathBuf},
            process::Command,
            sync::{Arc, Mutex, PoisonError},
            time::{Duration, Instant},
        };

        use openssl::{
            asn1::Asn1Time,
            ec::{EcGroup, EcKey},
            error::ErrorStack,
            hash::MessageDigest,
            nid::Nid,
            pkey::{PKey, Private},
            rsa::Rsa,
            ssl::{SslAcceptor, SslMethod, SslVerifyMode},
            x509::{
                X509, X509Builder, X509Name, X509NameBuilder, X509NameRef, X509Ref,
                extension::{BasicConstraints, SubjectAlternativeName},
            },
        };

        use litellm_secret_value::SecretValue;
        use rstest::rstest;

        use super::super::Pools;
        use crate::{Endpoint, PoolLimits, SslMode};

        const CHILD_DIRECTORY: &str = "LITELLM_DB_CLIENT_FILES_TEST_DIRECTORY";

        /// A certificate builder for `common_name`'s public key, issued by `issuer`, valid from today to tomorrow.
        fn builder(
            common_name: &str,
            key: &PKey<Private>,
            issuer: &X509NameRef,
        ) -> Result<X509Builder, ErrorStack> {
            let mut certificate = X509::builder()?;
            certificate.set_version(2)?;
            certificate.set_subject_name(&*name(common_name)?)?;
            certificate.set_issuer_name(issuer)?;
            certificate.set_pubkey(key)?;
            certificate.set_not_before(&*Asn1Time::days_from_now(0)?)?;
            certificate.set_not_after(&*Asn1Time::days_from_now(1)?)?;
            Ok(certificate)
        }

        fn name(common_name: &str) -> Result<X509Name, ErrorStack> {
            let mut name = X509NameBuilder::new()?;
            name.append_entry_by_nid(Nid::COMMONNAME, common_name)?;
            Ok(name.build())
        }

        fn self_signed(common_name: &str) -> Result<(X509, PKey<Private>), ErrorStack> {
            let key = PKey::from_rsa(Rsa::generate(2048)?)?;
            let mut certificate = builder(common_name, &key, &*name(common_name)?)?;
            certificate.append_extension(BasicConstraints::new().critical().ca().build()?)?;
            certificate.sign(&key, MessageDigest::sha256())?;
            Ok((certificate.build(), key))
        }

        /// A certificate for 127.0.0.1 issued by `root`.
        fn server_certificate(
            root: &X509Ref,
            root_key: &PKey<Private>,
        ) -> Result<(X509, PKey<Private>), ErrorStack> {
            let key = PKey::from_rsa(Rsa::generate(2048)?)?;
            let mut certificate = builder("127.0.0.1", &key, root.subject_name())?;
            let san = SubjectAlternativeName::new()
                .ip("127.0.0.1")
                .build(&certificate.x509v3_context(Some(root), None))?;
            certificate.append_extension(san)?;
            certificate.sign(root_key, MessageDigest::sha256())?;
            Ok((certificate.build(), key))
        }

        /// A client certificate with a SEC1 P-256 key, signed by `issuer_key` with RSA: both encode to the same
        /// length whatever the key, so a replacement keeps the files' sizes.
        fn same_size_client(
            common_name: &str,
            issuer_key: &PKey<Private>,
        ) -> Result<(Vec<u8>, Vec<u8>), ErrorStack> {
            let p256 = EcGroup::from_curve_name(Nid::X9_62_PRIME256V1)?;
            let key = EcKey::generate(&p256)?;
            let mut certificate = builder(
                common_name,
                &PKey::from_ec_key(key.clone())?,
                &*name("issuer")?,
            )?;
            certificate.sign(issuer_key, MessageDigest::sha256())?;
            Ok((certificate.build().to_pem()?, key.private_key_to_pem()?))
        }

        /// Puts a client certificate and its PKCS#1 key in place the way a mounted secret is replaced: new files
        /// renamed over the old ones.
        fn install(directory: &Path, common_name: &str) -> Result<(), Box<dyn StdError>> {
            let (certificate, key) = self_signed(common_name)?;
            std::fs::write(directory.join("cert.new"), certificate.to_pem()?)?;
            std::fs::write(directory.join("key.new"), key.rsa()?.private_key_to_pem()?)?;
            std::fs::rename(directory.join("cert.new"), directory.join("cert.pem"))?;
            std::fs::rename(directory.join("key.new"), directory.join("key.pem"))?;
            Ok(())
        }

        fn common_name(certificate: &X509Ref) -> Option<String> {
            let entry = certificate
                .subject_name()
                .entries_by_nid(Nid::COMMONNAME)
                .next()?;
            entry.data().to_string().ok()
        }

        /// Reads the body of a message that starts with its length.
        fn body(stream: &mut impl Read) -> std::io::Result<Vec<u8>> {
            let mut length = [0; 4];
            stream.read_exact(&mut length)?;
            let mut body = vec![0; (u32::from_be_bytes(length) as usize).saturating_sub(4)];
            stream.read_exact(&mut body)?;
            Ok(body)
        }

        /// Reads the startup message, asks for the password in cleartext and reads it.
        fn password(stream: &mut (impl Read + Write)) -> std::io::Result<String> {
            body(stream)?;
            stream.write_all(b"R\0\0\0\x08\0\0\0\x03")?;
            stream.read_exact(&mut [0])?;
            let password = body(stream)?;
            Ok(
                String::from_utf8_lossy(password.strip_suffix(b"\0").unwrap_or(&password))
                    .into_owned(),
            )
        }

        /// The common name of the client certificate and the password a connection presented.
        type Presented = (Option<String>, Option<String>);

        type Recorded = Arc<Mutex<Vec<Presented>>>;

        fn recording_server() -> Result<(u16, Recorded), Box<dyn StdError>> {
            let (certificate, key) = self_signed("server")?;
            recording_server_with(&certificate, &key)
        }

        /// A server that answers Postgres' SSLRequest, completes the TLS handshake with `certificate` asking for a
        /// client certificate, asks for the password, records the client certificate's common name and the
        /// password, and hangs up. A handshake the client gives up on is recorded as neither.
        fn recording_server_with(
            certificate: &X509Ref,
            key: &PKey<Private>,
        ) -> Result<(u16, Recorded), Box<dyn StdError>> {
            let mut acceptor = SslAcceptor::mozilla_intermediate_v5(SslMethod::tls())?;
            acceptor.set_certificate(certificate)?;
            acceptor.set_private_key(key)?;
            acceptor.set_verify_callback(SslVerifyMode::PEER, |_, _| true);
            let acceptor = acceptor.build();
            let listener = TcpListener::bind("127.0.0.1:0")?;
            let port = listener.local_addr()?.port();
            let presented = Arc::new(Mutex::new(Vec::new()));
            let recorded = Arc::clone(&presented);
            std::thread::spawn(move || {
                for mut socket in listener.incoming().flatten() {
                    if socket.read_exact(&mut [0; 8]).is_err() || socket.write_all(b"S").is_err() {
                        continue;
                    }
                    let mut tls = acceptor.accept(socket);
                    let seen = match &mut tls {
                        Ok(tls) => (
                            tls.ssl()
                                .peer_certificate()
                                .and_then(|certificate| common_name(&certificate)),
                            password(tls).ok(),
                        ),
                        Err(_) => (None, None),
                    };
                    recorded
                        .lock()
                        .unwrap_or_else(PoisonError::into_inner)
                        .push(seen);
                    drop(tls);
                }
            });
            Ok((port, presented))
        }

        /// Runs the calling test again in a child process whose environment sets each variable to a file in a new
        /// directory, since sqlx and OpenSSL read these variables from the environment. Returns the directory in the
        /// child, and None in the parent once the child passed.
        fn in_child(files: &[(&str, &str)]) -> Result<Option<PathBuf>, Box<dyn StdError>> {
            if let Some(directory) = std::env::var_os(CHILD_DIRECTORY) {
                return Ok(Some(PathBuf::from(directory)));
            }
            let test = std::thread::current()
                .name()
                .ok_or("the test thread is named")?
                .to_owned();
            let directory = std::env::temp_dir().join(format!(
                "litellm-db-client-files-{}-{}",
                std::process::id(),
                test.replace("::", "-")
            ));
            std::fs::create_dir_all(&directory)?;
            let mut child = Command::new(std::env::current_exe()?);
            child
                .args(["--exact", &test, "--nocapture"])
                .env(CHILD_DIRECTORY, &directory);
            for (variable, file) in files {
                child.env(variable, directory.join(file));
            }
            let output = child.output()?;
            std::fs::remove_dir_all(&directory)?;
            let stdout = String::from_utf8_lossy(&output.stdout);
            assert!(
                output.status.success() && stdout.contains("1 passed"),
                "{stdout}\n{}",
                String::from_utf8_lossy(&output.stderr)
            );
            Ok(None)
        }

        fn pools(port: u16, ssl_mode: SslMode) -> Result<Pools, crate::Error> {
            Pools::new(
                &Endpoint {
                    host: "127.0.0.1".to_owned(),
                    port,
                    socket: None,
                    username: "litellm".to_owned(),
                    password: Some(SecretValue::new("first")),
                    database: "litellm".to_owned(),
                    ssl_mode,
                    ssl_root_cert: None,
                    ssl_client: None,
                    application_name: None,
                    options: Vec::new(),
                    statement_cache_capacity: 0,
                    pool: PoolLimits {
                        max_connections: 1,
                        acquire_timeout_seconds: Some(5.0),
                        idle_timeout_seconds: None,
                        socket_timeout_seconds: None,
                    },
                },
                None,
            )
        }

        fn presented(recorded: &Recorded) -> Vec<Presented> {
            recorded
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .clone()
        }

        const CLIENT_FILES: &[(&str, &str)] = &[("PGSSLCERT", "cert.pem"), ("PGSSLKEY", "key.pem")];

        #[rstest]
        #[tokio::test]
        async fn new_connections_present_the_client_files_as_they_are_now()
        -> Result<(), Box<dyn StdError>> {
            let Some(directory) = in_child(CLIENT_FILES)? else {
                return Ok(());
            };
            let (port, presented_to_server) = recording_server()?;
            let pools = pools(port, SslMode::Require)?;

            install(&directory, "mounted-after-startup")?;
            let first = pools.writer_only().await;
            pools.set_writer_password(SecretValue::new("rotated"));
            install(&directory, "replaced")?;
            let second = pools.writer_only().await;

            assert!(
                first.is_err() && second.is_err(),
                "the server hangs up after the password"
            );
            let client =
                |name: &str, password: &str| (Some(name.to_owned()), Some(password.to_owned()));
            assert_eq!(
                presented(&presented_to_server),
                [
                    client("mounted-after-startup", "first"),
                    client("replaced", "rotated")
                ]
            );
            Ok(())
        }

        #[rstest]
        #[tokio::test]
        async fn new_connections_see_a_replacement_that_keeps_the_files_times_and_sizes()
        -> Result<(), Box<dyn StdError>> {
            let Some(directory) = in_child(CLIENT_FILES)? else {
                return Ok(());
            };
            let (port, presented_to_server) = recording_server()?;
            let issuer = PKey::from_rsa(Rsa::generate(2048)?)?;
            let (cert, key) = (directory.join("cert.pem"), directory.join("key.pem"));
            let (first_cert, first_key) = same_size_client("first-client", &issuer)?;
            std::fs::write(&cert, first_cert)?;
            std::fs::write(&key, first_key)?;
            let written = |path: &Path| std::fs::metadata(path).and_then(|file| file.modified());
            let (cert_written, key_written) = (written(&cert)?, written(&key)?);
            let sizes = || -> std::io::Result<_> {
                Ok((
                    std::fs::metadata(&cert)?.len(),
                    std::fs::metadata(&key)?.len(),
                ))
            };
            let first_sizes = sizes()?;
            let pools = pools(port, SslMode::Require)?;
            let first = pools.writer_only().await;

            // The system moves a file's change time on a clock that can tick only every few milliseconds.
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
            let (second_cert, second_key) = same_size_client("other-client", &issuer)?;
            std::fs::write(&cert, second_cert)?;
            std::fs::write(&key, second_key)?;
            std::fs::File::options()
                .write(true)
                .open(&cert)?
                .set_modified(cert_written)?;
            std::fs::File::options()
                .write(true)
                .open(&key)?
                .set_modified(key_written)?;
            assert_eq!(sizes()?, first_sizes);
            assert_eq!(
                (written(&cert)?, written(&key)?),
                (cert_written, key_written)
            );
            let second = pools.writer_only().await;

            assert!(
                first.is_err() && second.is_err(),
                "the server hangs up after the password"
            );
            let client = |name: &str| (Some(name.to_owned()), Some("first".to_owned()));
            assert_eq!(
                presented(&presented_to_server),
                [client("first-client"), client("other-client")]
            );
            Ok(())
        }

        const ROOT_FILES: &[(&str, &str)] = &[
            ("PGSSLROOTCERT", "root.crt"),
            ("SSL_CERT_FILE", "system.pem"),
        ];

        /// Whether the server's certificate was accepted on each of the first `count` connections: the server then
        /// got as far as reading the password. A client that refuses the certificate can return before the server
        /// records the handshake it gave up on, so this waits until the server recorded `count` connections.
        fn accepted(recorded: &Recorded, count: usize) -> Vec<bool> {
            let deadline = Instant::now() + Duration::from_secs(10);
            while presented(recorded).len() < count && Instant::now() < deadline {
                std::thread::sleep(Duration::from_millis(10));
            }
            presented(recorded)
                .into_iter()
                .map(|(_, password)| password.is_some())
                .collect()
        }

        #[rstest]
        #[case::der_root_with_the_issuer_among_the_system_roots(true, true, true)]
        #[case::der_root_without_the_issuer_among_the_system_roots(true, false, false)]
        #[case::pem_root(false, false, true)]
        #[tokio::test]
        async fn verify_full_checks_the_server_the_way_rustls_does_with_a_root_cert_that_is_not_pem(
            #[case] der: bool,
            #[case] issuer_among_system_roots: bool,
            #[case] accepted_server: bool,
        ) -> Result<(), Box<dyn StdError>> {
            let Some(directory) = in_child(ROOT_FILES)? else {
                return Ok(());
            };
            let (root, root_key) = self_signed("database-root")?;
            let (other_root, _) = self_signed("other-root")?;
            std::fs::write(
                directory.join("root.crt"),
                if der { root.to_der()? } else { root.to_pem()? },
            )?;
            std::fs::write(
                directory.join("system.pem"),
                if issuer_among_system_roots {
                    &root
                } else {
                    &other_root
                }
                .to_pem()?,
            )?;
            let (certificate, key) = server_certificate(&root, &root_key)?;
            let (port, presented_to_server) = recording_server_with(&certificate, &key)?;

            let connected = pools(port, SslMode::VerifyFull)?.writer_only().await;

            assert!(connected.is_err(), "the server hangs up after the password");
            assert_eq!(accepted(&presented_to_server, 1), [accepted_server]);
            Ok(())
        }

        #[rstest]
        #[tokio::test]
        async fn new_connections_use_a_root_cert_that_turned_into_pem()
        -> Result<(), Box<dyn StdError>> {
            let Some(directory) = in_child(ROOT_FILES)? else {
                return Ok(());
            };
            let (root, root_key) = self_signed("database-root")?;
            let (other_root, _) = self_signed("other-root")?;
            let root_file = directory.join("root.crt");
            std::fs::write(&root_file, root.to_der()?)?;
            std::fs::write(directory.join("system.pem"), other_root.to_pem()?)?;
            let (certificate, key) = server_certificate(&root, &root_key)?;
            let (port, presented_to_server) = recording_server_with(&certificate, &key)?;
            let pools = pools(port, SslMode::VerifyFull)?;
            let first = pools.writer_only().await;

            std::fs::write(directory.join("root.new"), root.to_pem()?)?;
            std::fs::rename(directory.join("root.new"), &root_file)?;
            let second = pools.writer_only().await;

            assert!(
                first.is_err() && second.is_err(),
                "the server hangs up after the password"
            );
            assert_eq!(accepted(&presented_to_server, 2), [false, true]);
            Ok(())
        }
    }
}
