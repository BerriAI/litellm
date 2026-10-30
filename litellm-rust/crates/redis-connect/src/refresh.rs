use std::{
    future::Future,
    pin::Pin,
    sync::{Arc, Mutex},
    time::{Duration, SystemTime},
};

use futures_util::{Stream, stream};
use litellm_redis_client::{BasicAuth, RedisError, StreamingCredentialsProvider};
use tokio::sync::watch;

const TOKEN_ONLY_USERNAME: &str = "default";
const UNKNOWN_LIFETIME: Duration = Duration::from_secs(300);
const MIN_REFRESH: Duration = Duration::from_secs(1);
const FIRST_RETRY: Duration = Duration::from_secs(1);
const MAX_RETRY: Duration = Duration::from_secs(60);

#[derive(Clone, PartialEq, Eq)]
pub struct Expiring {
    pub username: Option<String>,
    pub password: String,
    pub expires_at: Option<SystemTime>,
}

pub type SourceError = Box<dyn std::error::Error + Send + Sync>;

pub type SourceFuture<'a> = Pin<Box<dyn Future<Output = Result<Expiring, SourceError>> + Send + 'a>>;

pub trait CredentialSource: Send + Sync + 'static {
    fn fetch(&self) -> SourceFuture<'_>;
}

#[derive(Clone)]
enum Latest {
    Pending,
    Ready(BasicAuth),
    Unavailable(Arc<str>),
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
        let start = self.start.lock().unwrap_or_else(|poisoned| poisoned.into_inner()).take();
        if let Some(start) = start {
            start();
        }
    }
}

impl StreamingCredentialsProvider for RefreshingCredentials {
    fn subscribe(&self) -> Pin<Box<dyn Stream<Item = Result<BasicAuth, RedisError>> + Send + 'static>> {
        self.start_once();
        let receiver = self.latest.clone();
        Box::pin(stream::unfold((receiver, true), |(mut receiver, first)| async move {
            loop {
                let changed = match first {
                    true => Ok(()),
                    false => receiver.changed().await,
                };
                changed.ok()?;
                let latest = receiver.borrow_and_update().clone();
                match latest {
                    Latest::Pending => {
                        receiver.changed().await.ok()?;
                        receiver.mark_changed();
                    }
                    Latest::Ready(credentials) => return Some((Ok(credentials), (receiver, false))),
                    Latest::Unavailable(reason) => {
                        let error = RedisError::from((
                            litellm_redis_client::RedisErrorKind::AuthenticationFailed,
                            "fetching Redis credentials failed",
                            reason.to_string(),
                        ));
                        return Some((Err(error), (receiver, false)));
                    }
                }
            }
        }))
    }
}

async fn refresh(source: impl CredentialSource, sender: watch::Sender<Latest>) {
    let mut retry = FIRST_RETRY;
    loop {
        let wait = match source.fetch().await {
            Ok(credentials) => {
                retry = FIRST_RETRY;
                let wait = refresh_after(credentials.expires_at, SystemTime::now());
                sender.send_replace(Latest::Ready(BasicAuth::new(
                    credentials.username.unwrap_or_else(|| TOKEN_ONLY_USERNAME.to_owned()),
                    credentials.password,
                )));
                wait
            }
            Err(error) => {
                tracing::warn!(error = %error, retry_in = ?retry, "fetching Redis credentials failed");
                if !matches!(*sender.borrow(), Latest::Ready(_)) {
                    sender.send_replace(Latest::Unavailable(error.to_string().into()));
                }
                let wait = retry;
                retry = (retry * 2).min(MAX_RETRY);
                wait
            }
        };
        tokio::select! {
            () = sender.closed() => return,
            () = tokio::time::sleep(wait) => {}
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
        sync::{
            Arc, Mutex,
            atomic::{AtomicUsize, Ordering},
        },
        time::{Duration, SystemTime},
    };

    use futures_util::StreamExt;
    use litellm_redis_client::StreamingCredentialsProvider;
    use rstest::rstest;

    use super::{CredentialSource, Expiring, RefreshingCredentials, SourceFuture, refresh_after};

    #[rstest]
    #[case::three_quarters_of_the_lifetime(Some(Duration::from_secs(900)), Duration::from_secs(675))]
    #[case::already_expired(Some(Duration::ZERO), Duration::from_secs(1))]
    #[case::unknown_expiry(None, Duration::from_secs(225))]
    fn refreshes_well_before_expiry(#[case] lifetime: Option<Duration>, #[case] expected: Duration) {
        let now = SystemTime::UNIX_EPOCH + Duration::from_secs(1_000);

        assert_eq!(refresh_after(lifetime.map(|lifetime| now + lifetime), now), expected);
    }

    struct Scripted {
        replies: Mutex<Vec<Result<Expiring, &'static str>>>,
        calls: Arc<AtomicUsize>,
    }

    impl CredentialSource for Scripted {
        fn fetch(&self) -> SourceFuture<'_> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            let reply = self.replies.lock().unwrap().remove(0);
            Box::pin(async move { reply.map_err(Into::into) })
        }
    }

    fn token(password: &str, lifetime: Duration) -> Result<Expiring, &'static str> {
        Ok(Expiring {
            username: None,
            password: password.to_owned(),
            expires_at: Some(SystemTime::now() + lifetime),
        })
    }

    #[rstest]
    #[tokio::test(start_paused = true)]
    async fn pushes_each_refresh_and_keeps_the_last_token_through_failures() {
        let calls = Arc::new(AtomicUsize::new(0));
        let provider = RefreshingCredentials::new(Scripted {
            replies: Mutex::new(vec![
                token("first", Duration::from_secs(4)),
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
        let runtime = tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap();

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
            replies: Mutex::new(vec![Err("no identity"), token("late", Duration::from_secs(3600))]),
            calls: Arc::new(AtomicUsize::new(0)),
        });
        let mut stream = provider.subscribe();

        assert!(stream.next().await.unwrap().is_err());
        assert_eq!(stream.next().await.unwrap().unwrap().password(), "late");
    }
}
