use std::{
    future::{Future, pending},
    pin::Pin,
    sync::{Arc, Mutex},
    time::{Duration, SystemTime},
};

use futures_util::{Stream, stream};
use litellm_redis_client::{BasicAuth, RedisError, StreamingCredentialsProvider};
use litellm_secret_value::SecretValue;
use tokio::{sync::watch, time::Instant};

use crate::Error;

const TOKEN_ONLY_USERNAME: &str = "default";
const UNKNOWN_LIFETIME: Duration = Duration::from_secs(300);
const MIN_REFRESH: Duration = Duration::from_secs(1);
const ACQUISITION_TIMEOUT: Duration = Duration::from_secs(30);
const FIRST_RETRY: Duration = Duration::from_secs(1);
const MAX_RETRY: Duration = Duration::from_secs(60);

#[derive(Clone, PartialEq, Eq)]
pub struct Expiring {
    pub username: Option<String>,
    pub password: SecretValue,
    pub expires_at: Option<SystemTime>,
}

pub type SourceFuture<'a> = Pin<Box<dyn Future<Output = Result<Expiring, Error>> + Send + 'a>>;

pub trait CredentialSource: Send + Sync + 'static {
    fn fetch(&self) -> SourceFuture<'_>;
}

#[derive(Clone)]
enum Latest {
    Pending,
    Ready {
        username: String,
        password: SecretValue,
    },
    Unavailable(Arc<Error>),
}

impl Latest {
    fn is_same_token(&self, other: &Self) -> bool {
        matches!(
            (self, other),
            (
                Self::Ready { username, password },
                Self::Ready { username: other_username, password: other_password },
            ) if username == other_username && password == other_password
        )
    }
}

type Start = Box<dyn FnOnce() + Send>;

#[derive(Clone)]
pub struct RefreshingCredentials {
    latest: watch::Receiver<Latest>,
    start: Arc<Mutex<Option<Start>>>,
}

impl RefreshingCredentials {
    pub fn new(source: impl CredentialSource) -> Self {
        let (sender, latest) = watch::channel(Latest::Pending);
        let start: Start = Box::new(move || drop(tokio::spawn(refresh(source, sender))));
        Self {
            latest,
            start: Arc::new(Mutex::new(Some(start))),
        }
    }

    fn start_once(&self) {
        let start = self
            .start
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .take();
        if let Some(start) = start {
            start();
        }
    }
}

impl StreamingCredentialsProvider for RefreshingCredentials {
    fn subscribe(
        &self,
    ) -> Pin<Box<dyn Stream<Item = Result<BasicAuth, RedisError>> + Send + 'static>> {
        self.start_once();
        Box::pin(stream::unfold(
            (self.latest.clone(), true),
            |(mut receiver, first)| async move {
                if !first {
                    receiver.changed().await.ok()?;
                }
                let latest = receiver
                    .wait_for(|latest| !matches!(latest, Latest::Pending))
                    .await
                    .ok()?
                    .clone();
                let item = match latest {
                    Latest::Ready { username, password } => {
                        Ok(BasicAuth::new(username, password.expose().to_owned()))
                    }
                    Latest::Unavailable(error) => Err(public_error(&error)),
                    Latest::Pending => unreachable!("wait_for skips Pending"),
                };
                Some((item, (receiver, false)))
            },
        ))
    }
}

fn public_error(error: &Error) -> RedisError {
    RedisError::from((
        litellm_redis_client::RedisErrorKind::AuthenticationFailed,
        "fetching Redis credentials failed",
        error.to_string(),
    ))
}

struct Fresh {
    latest: Latest,
    expires_at: Option<Instant>,
    refresh_in: Duration,
}

fn fresh(credentials: Expiring, now: SystemTime) -> Result<Fresh, Error> {
    let remaining = credentials
        .expires_at
        .map(|expires_at| expires_at.duration_since(now).unwrap_or_default());
    if remaining == Some(Duration::ZERO) {
        return Err(Error::ExpiredCredential);
    }
    Ok(Fresh {
        latest: Latest::Ready {
            username: credentials
                .username
                .unwrap_or_else(|| TOKEN_ONLY_USERNAME.to_owned()),
            password: credentials.password,
        },
        expires_at: remaining.map(|remaining| Instant::now() + remaining),
        refresh_in: refresh_after(credentials.expires_at, now),
    })
}

async fn until_expiry<T>(
    future: impl Future<Output = T>,
    sender: &watch::Sender<Latest>,
    expires_at: Option<Instant>,
) -> Option<T> {
    let expiration = async {
        match expires_at {
            Some(deadline) => tokio::time::sleep_until(deadline).await,
            None => pending().await,
        }
    };
    tokio::pin!(future, expiration);
    let mut expired = false;
    loop {
        tokio::select! {
            biased;
            () = sender.closed() => return None,
            () = &mut expiration, if !expired => {
                expired = true;
                sender.send_replace(Latest::Unavailable(Arc::new(Error::ExpiredCredential)));
            }
            result = &mut future => return Some(result),
        }
    }
}

async fn refresh(source: impl CredentialSource, sender: watch::Sender<Latest>) {
    let mut retry = FIRST_RETRY;
    let mut expires_at = None;
    loop {
        let acquisition = async {
            tokio::time::timeout(ACQUISITION_TIMEOUT, source.fetch())
                .await
                .map_err(|cause| Error::CredentialTimeout {
                    timeout: ACQUISITION_TIMEOUT,
                    cause,
                })?
                .and_then(|credentials| fresh(credentials, SystemTime::now()))
        };
        let Some(result) = until_expiry(acquisition, &sender, expires_at).await else {
            return;
        };
        let wait = match result {
            Ok(Fresh {
                latest,
                expires_at: deadline,
                refresh_in,
            }) => {
                retry = FIRST_RETRY;
                expires_at = deadline;
                sender.send_if_modified(|current| {
                    let changed = !current.is_same_token(&latest);
                    if changed {
                        *current = latest;
                    }
                    changed
                });
                refresh_in
            }
            Err(error) => {
                tracing::warn!(failure = error.kind(), upstream_status = error.upstream_status(), retry_in = ?retry, "fetching Redis credentials failed");
                if !matches!(*sender.borrow(), Latest::Ready { .. }) {
                    sender.send_replace(Latest::Unavailable(Arc::new(error)));
                }
                let wait = retry;
                retry = (retry * 2).min(MAX_RETRY);
                wait
            }
        };
        if until_expiry(tokio::time::sleep(wait), &sender, expires_at)
            .await
            .is_none()
        {
            return;
        }
    }
}

fn refresh_after(expires_at: Option<SystemTime>, now: SystemTime) -> Duration {
    let remaining = expires_at
        .map(|expires_at| expires_at.duration_since(now).unwrap_or_default())
        .unwrap_or(UNKNOWN_LIFETIME);
    (remaining * 3 / 4).max(MIN_REFRESH)
}

#[cfg(test)]
mod tests {
    use std::{
        collections::BTreeSet,
        sync::{
            Arc, Mutex,
            atomic::{AtomicUsize, Ordering},
        },
        time::{Duration, SystemTime},
    };

    use futures_util::StreamExt;
    use litellm_redis_client::StreamingCredentialsProvider;
    use litellm_secret_value::SecretValue;
    use rstest::rstest;

    use super::{CredentialSource, Expiring, RefreshingCredentials, SourceFuture, refresh_after};
    use litellm_redis_client::{
        ClusterEndpoint, ClusterNode, Credentials, CredentialsProvider, Endpoint, Namespace,
        RedisClient, RedisConfig, Timeouts, TlsOptions,
    };
    use litellm_redis_testing::{ClientConnection, RedisDeployment, Topology, eventually};

    #[rstest]
    #[case::three_quarters_of_the_lifetime(
        Some(Duration::from_secs(900)),
        Duration::from_secs(675)
    )]
    #[case::already_expired(Some(Duration::ZERO), Duration::from_secs(1))]
    #[case::unknown_expiry(None, Duration::from_secs(225))]
    fn refreshes_well_before_expiry(
        #[case] lifetime: Option<Duration>,
        #[case] expected: Duration,
    ) {
        let now = SystemTime::UNIX_EPOCH + Duration::from_secs(1_000);

        assert_eq!(
            refresh_after(lifetime.map(|lifetime| now + lifetime), now),
            expected
        );
    }

    struct Scripted {
        replies: Mutex<Vec<Result<Expiring, &'static str>>>,
        calls: Arc<AtomicUsize>,
    }

    impl CredentialSource for Scripted {
        fn fetch(&self) -> SourceFuture<'_> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            let reply = self.replies.lock().unwrap().remove(0);
            Box::pin(async move {
                reply.map_err(|reason| {
                    crate::Error::credential(crate::error::CredentialStage::Identity)(
                        std::io::Error::other(reason),
                    )
                })
            })
        }
    }

    fn token(password: &str, lifetime: Duration) -> Result<Expiring, &'static str> {
        Ok(Expiring {
            username: None,
            password: SecretValue::new(password),
            expires_at: Some(SystemTime::now() + lifetime),
        })
    }

    #[rstest]
    #[tokio::test(start_paused = true)]
    async fn pushes_each_refresh_and_keeps_the_last_token_through_failures() {
        let calls = Arc::new(AtomicUsize::new(0));
        let provider = RefreshingCredentials::new(Scripted {
            replies: Mutex::new(vec![
                token("first", Duration::from_secs(8)),
                Err("token endpoint down"),
                token("second", Duration::from_secs(3600)),
            ]),
            calls: calls.clone(),
        });
        let mut stream = provider.subscribe();

        let first = stream.next().await.unwrap().unwrap();
        let second = stream.next().await.unwrap().unwrap();
        let late = provider.subscribe().next().await.unwrap().unwrap();

        assert_eq!((first.username(), first.password()), ("default", "first"));
        assert_eq!(second.password(), "second");
        assert_eq!(late.password(), "second");
        assert_eq!(calls.load(Ordering::SeqCst), 3);
    }

    #[rstest]
    fn building_outside_a_runtime_fetches_nothing_until_a_connection_subscribes() {
        let calls = Arc::new(AtomicUsize::new(0));
        let provider = RefreshingCredentials::new(Scripted {
            replies: Mutex::new(vec![token("first", Duration::from_secs(3600))]),
            calls: calls.clone(),
        });
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();

        let before = calls.load(Ordering::SeqCst);
        let first = runtime.block_on(async { provider.subscribe().next().await.unwrap().unwrap() });

        assert_eq!(before, 0);
        assert_eq!(first.password(), "first");
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }

    #[rstest]
    #[tokio::test(start_paused = true)]
    async fn a_failure_before_any_token_fails_the_connection_instead_of_hanging() {
        let provider = RefreshingCredentials::new(Scripted {
            replies: Mutex::new(vec![
                Err("no identity"),
                token("late", Duration::from_secs(3600)),
            ]),
            calls: Arc::new(AtomicUsize::new(0)),
        });
        let mut stream = provider.subscribe();

        assert!(stream.next().await.unwrap().is_err());
        assert_eq!(stream.next().await.unwrap().unwrap().password(), "late");
    }

    struct Blocking {
        first: Mutex<Option<Expiring>>,
        started: Arc<tokio::sync::Notify>,
        cancelled: Arc<AtomicUsize>,
    }

    struct Cancellation(Arc<AtomicUsize>);

    impl Drop for Cancellation {
        fn drop(&mut self) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }

    impl CredentialSource for Blocking {
        fn fetch(&self) -> SourceFuture<'_> {
            let first = self.first.lock().unwrap().take();
            Box::pin(async move {
                if let Some(first) = first {
                    return Ok(first);
                }
                let _guard = Cancellation(self.cancelled.clone());
                self.started.notify_one();
                std::future::pending().await
            })
        }
    }

    #[rstest]
    #[tokio::test(start_paused = true)]
    async fn a_refresh_that_returns_the_same_token_is_not_pushed_again() {
        let calls = Arc::new(AtomicUsize::new(0));
        let provider = RefreshingCredentials::new(Scripted {
            replies: Mutex::new(vec![
                token("same", Duration::from_secs(8)),
                token("same", Duration::from_secs(3600)),
            ]),
            calls: calls.clone(),
        });
        let mut stream = provider.subscribe();
        assert_eq!(stream.next().await.unwrap().unwrap().password(), "same");

        let pushed = tokio::time::timeout(Duration::from_secs(60), stream.next()).await;

        assert_eq!(calls.load(Ordering::SeqCst), 2);
        assert!(pushed.is_err(), "an unchanged token was pushed again");
    }

    #[rstest]
    #[tokio::test(start_paused = true)]
    async fn expiration_is_published_while_refresh_is_stalled() {
        let started = Arc::new(tokio::sync::Notify::new());
        let provider = RefreshingCredentials::new(Blocking {
            first: Mutex::new(Some(token("first", Duration::from_secs(4)).unwrap())),
            started: started.clone(),
            cancelled: Arc::new(AtomicUsize::new(0)),
        });
        let mut stream = provider.subscribe();
        assert_eq!(stream.next().await.unwrap().unwrap().password(), "first");
        let start = tokio::time::Instant::now();
        started.notified().await;
        let expired = stream.next().await.unwrap().unwrap_err();
        assert!(start.elapsed() <= Duration::from_secs(5));
        assert!(expired.to_string().contains("expired"));
        let fresh = provider.subscribe().next().await.unwrap().unwrap_err();
        assert!(fresh.to_string().contains("expired"));
    }

    #[rstest]
    #[tokio::test(start_paused = true)]
    async fn initial_acquisition_has_a_bounded_wait() {
        let provider = RefreshingCredentials::new(Blocking {
            first: Mutex::new(None),
            started: Arc::new(tokio::sync::Notify::new()),
            cancelled: Arc::new(AtomicUsize::new(0)),
        });
        let error = provider.subscribe().next().await.unwrap().unwrap_err();
        assert!(error.to_string().contains("exceeded"));
    }

    #[rstest]
    #[tokio::test(start_paused = true)]
    async fn dropping_all_subscribers_cancels_inflight_acquisition() {
        let started = Arc::new(tokio::sync::Notify::new());
        let cancelled = Arc::new(AtomicUsize::new(0));
        let provider = RefreshingCredentials::new(Blocking {
            first: Mutex::new(None),
            started: started.clone(),
            cancelled: cancelled.clone(),
        });
        let stream = provider.subscribe();
        started.notified().await;
        drop(stream);
        drop(provider);
        tokio::task::yield_now().await;
        assert_eq!(cancelled.load(Ordering::SeqCst), 1);
    }

    #[rstest]
    #[tokio::test(start_paused = true)]
    async fn an_expired_fetch_is_rejected_and_a_later_valid_token_recovers() {
        let provider = RefreshingCredentials::new(Scripted {
            replies: Mutex::new(vec![
                Ok(Expiring {
                    username: None,
                    password: SecretValue::new("expired"),
                    expires_at: Some(SystemTime::UNIX_EPOCH),
                }),
                token("recovered", Duration::from_secs(3600)),
            ]),
            calls: Arc::new(AtomicUsize::new(0)),
        });
        let mut stream = provider.subscribe();
        assert!(
            stream
                .next()
                .await
                .unwrap()
                .unwrap_err()
                .to_string()
                .contains("expired")
        );
        assert_eq!(
            stream.next().await.unwrap().unwrap().password(),
            "recovered"
        );
    }

    #[rstest]
    fn credential_failure_keeps_its_typed_cause_but_hides_it_at_the_redis_boundary() {
        let error = crate::Error::credential(crate::error::CredentialStage::Identity)(
            std::io::Error::new(std::io::ErrorKind::PermissionDenied, "credential-value"),
        );
        assert_eq!(
            litellm_error::find::<std::io::Error>(&error)
                .unwrap()
                .kind(),
            std::io::ErrorKind::PermissionDenied
        );
        let debug = format!("{error:?}");
        assert!(!debug.contains("credential-value"));
        assert!(debug.contains("[REDACTED]"));
        let public = super::public_error(&error).to_string();
        assert!(public.contains("caller identity"));
        assert!(!public.contains("credential-value"));
    }

    #[rstest]
    fn credential_http_failures_expose_status_without_exposing_the_response_body() {
        let error = crate::Error::credential_response(crate::CredentialStage::Exchange, 503)(
            litellm_http::Error::upstream(503, "credential-value".into()),
        );
        let public = super::public_error(&error).to_string();
        assert!(public.contains("HTTP 503"));
        assert!(!public.contains("credential-value"));
        assert_eq!(
            litellm_error::find::<litellm_http::Error>(&error)
                .unwrap()
                .upstream_response()
                .unwrap()
                .0,
            503
        );
    }

    fn endpoint(deployment: &RedisDeployment) -> Endpoint {
        match deployment {
            RedisDeployment::Standalone(server) => Endpoint::Url {
                url: SecretValue::new(server.url()),
                tls: TlsOptions::verified(),
            },
            RedisDeployment::Cluster(cluster) => Endpoint::Cluster(ClusterEndpoint {
                nodes: cluster
                    .nodes()
                    .map(|(host, port)| ClusterNode {
                        host: host.to_owned(),
                        port,
                    })
                    .collect(),
                tls: None,
            }),
        }
    }

    fn user(username: &str, password: &str, lifetime: Duration) -> Result<Expiring, &'static str> {
        Ok(Expiring {
            username: Some(username.to_owned()),
            password: SecretValue::new(password),
            expires_at: Some(SystemTime::now() + lifetime),
        })
    }

    fn connection_users(connections: &[ClientConnection]) -> BTreeSet<&str> {
        connections
            .iter()
            .map(|connection| connection.user.as_str())
            .collect()
    }

    fn connection_ids(connections: &[ClientConnection]) -> BTreeSet<(u16, u64)> {
        connections
            .iter()
            .map(|connection| (connection.node, connection.id))
            .collect()
    }

    #[rstest]
    #[tokio::test]
    async fn a_refreshed_token_reauthenticates_live_connections_before_the_old_one_expires(
        #[values(Topology::Standalone, Topology::Cluster)] topology: Topology,
    ) {
        let server = RedisDeployment::start(topology).await.unwrap();
        server
            .cli_on_every_node(&[
                "ACL", "SETUSER", "alice", "on", ">first", "~*", "&*", "+@all",
            ])
            .await
            .unwrap();
        server
            .cli_on_every_node(&[
                "ACL", "SETUSER", "bob", "on", ">second", "~*", "&*", "+@all",
            ])
            .await
            .unwrap();
        let provider = RefreshingCredentials::new(Scripted {
            replies: Mutex::new(vec![
                user("alice", "first", Duration::from_secs(2)),
                user("bob", "second", Duration::from_secs(3600)),
            ]),
            calls: Arc::new(AtomicUsize::new(0)),
        });
        let client = RedisClient::new(RedisConfig {
            endpoint: endpoint(&server),
            credentials: Credentials::Provider(CredentialsProvider(Arc::new(provider))),
            namespace: Namespace::default(),
            timeouts: Timeouts::default(),
            breaker: None,
            events: None,
        })
        .unwrap();
        client.set("k", "1", Duration::from_secs(60)).await.unwrap();
        let before = server.clients().await.unwrap();

        let after = eventually(async || {
            let connections = server.clients().await.ok()?;
            (!connections.is_empty() && connection_users(&connections) == BTreeSet::from(["bob"]))
                .then_some(connections)
        })
        .await;
        server
            .cli_on_every_node(&["ACL", "DELUSER", "alice"])
            .await
            .unwrap();

        assert!(!before.is_empty());
        assert_eq!(connection_users(&before), BTreeSet::from(["alice"]));
        assert_eq!(connection_ids(&after), connection_ids(&before));
        assert_eq!(
            client.get::<String>("k").await.unwrap().as_deref(),
            Some("1")
        );
        server
            .cli_on_every_node(&["CLIENT", "KILL", "USER", "bob"])
            .await
            .unwrap();
        let value = eventually(async || client.get::<String>("k").await.ok().flatten()).await;
        let reconnected = server.clients().await.unwrap();

        assert_eq!(value, "1");
        assert!(!reconnected.is_empty());
        assert_eq!(connection_users(&reconnected), BTreeSet::from(["bob"]));
        assert!(connection_ids(&reconnected).is_disjoint(&connection_ids(&after)));
    }
}
