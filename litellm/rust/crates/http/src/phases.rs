use std::{
    future::Future,
    pin::Pin,
    sync::{Arc, Mutex, MutexGuard, PoisonError},
    task::{Context, Poll},
    time::Duration,
};

use tokio::{sync::Notify, time::Instant};
use tower::{Layer, Service, timeout::error::Elapsed};

use crate::{Client, Error};

type BoxError = Box<dyn std::error::Error + Send + Sync>;

tokio::task_local! {
    static CONNECTING: Arc<Connection>;
}

/// httpx's per-phase limits: `connect` bounds opening a new connection and `read` bounds the
/// wait for the response head once no connection is being opened.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Phases {
    pub connect: Duration,
    pub read: Duration,
}

#[derive(Default)]
struct Opening {
    in_flight: usize,
    settled_at: Option<Instant>,
}

struct Connection {
    bound: Duration,
    opening: Mutex<Opening>,
    changed: Notify,
}

impl Connection {
    fn lock(&self) -> MutexGuard<'_, Opening> {
        self.opening.lock().unwrap_or_else(PoisonError::into_inner)
    }

    fn reading_since(&self, sent: Instant) -> Option<Instant> {
        let opening = self.lock();
        (opening.in_flight == 0).then(|| opening.settled_at.unwrap_or(sent))
    }
}

struct InFlight(Arc<Connection>);

impl InFlight {
    fn start(connection: Arc<Connection>) -> Self {
        connection.lock().in_flight += 1;
        connection.changed.notify_waiters();
        Self(connection)
    }
}

impl Drop for InFlight {
    fn drop(&mut self) {
        {
            let mut opening = self.0.lock();
            opening.in_flight -= 1;
            opening.settled_at = Some(Instant::now());
        }
        self.0.changed.notify_waiters();
    }
}

/// Bounds each new connection by the calling request's [`Phases::connect`], or by `default`
/// outside [`Client::execute_in_phases`].
#[derive(Clone, Copy, Debug)]
pub(crate) struct ConnectBound {
    default: Duration,
}

impl ConnectBound {
    pub(crate) fn new(default: Duration) -> Self {
        Self { default }
    }
}

impl<S> Layer<S> for ConnectBound {
    type Service = ConnectBounded<S>;

    fn layer(&self, inner: S) -> Self::Service {
        ConnectBounded {
            inner,
            default: self.default,
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) struct ConnectBounded<S> {
    inner: S,
    default: Duration,
}

impl<S, R> Service<R> for ConnectBounded<S>
where
    S: Service<R, Error = BoxError>,
    S::Future: Send + 'static,
    S::Response: 'static,
{
    type Response = S::Response;
    type Error = BoxError;
    type Future = Pin<Box<dyn Future<Output = Result<S::Response, BoxError>> + Send>>;

    fn poll_ready(&mut self, cx: &mut Context<'_>) -> Poll<Result<(), BoxError>> {
        self.inner.poll_ready(cx)
    }

    fn call(&mut self, request: R) -> Self::Future {
        let connection = CONNECTING.try_with(Arc::clone).ok();
        let bound = connection
            .as_ref()
            .map_or(self.default, |connection| connection.bound);
        let in_flight = connection.map(InFlight::start);
        let connecting = self.inner.call(request);
        Box::pin(async move {
            let connected = tokio::time::timeout(bound, connecting).await;
            drop(in_flight);
            connected.map_err(|_| Box::new(Elapsed::new()) as BoxError)?
        })
    }
}

impl Client {
    /// Sends `request` with httpx's phase limits: a new connection that takes longer than
    /// `phases.connect` fails as a connect timeout, and the response head then has
    /// `phases.read` from when the connection was ready.
    pub async fn execute_in_phases(
        &self,
        request: reqwest::Request,
        phases: Phases,
    ) -> Result<reqwest::Response, Error> {
        let connection = Arc::new(Connection {
            bound: phases.connect,
            opening: Mutex::default(),
            changed: Notify::new(),
        });
        let sent = Instant::now();
        let response = CONNECTING.scope(Arc::clone(&connection), self.execute(request));
        tokio::pin!(response);
        loop {
            let changed = connection.changed.notified();
            tokio::pin!(changed);
            changed.as_mut().enable();
            let Some(since) = connection.reading_since(sent) else {
                tokio::select! {
                    response = &mut response => return response.map_err(Error::from),
                    () = &mut changed => continue,
                }
            };
            tokio::select! {
                response = tokio::time::timeout_at(since + phases.read, &mut response) => {
                    return response.map_err(Error::read_timeout)?.map_err(Error::from);
                }
                () = &mut changed => {}
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::net::SocketAddr;

    use rstest::rstest;
    use tokio::{
        io::{AsyncReadExt, AsyncWriteExt},
        net::TcpListener,
    };

    use super::*;

    const LIMIT: Duration = Duration::from_millis(400);

    #[derive(Clone)]
    struct SlowConnect(Duration);

    impl<S> Layer<S> for SlowConnect {
        type Service = SlowConnected<S>;

        fn layer(&self, inner: S) -> Self::Service {
            SlowConnected(inner, self.0)
        }
    }

    #[derive(Clone)]
    struct SlowConnected<S>(S, Duration);

    impl<S, R> Service<R> for SlowConnected<S>
    where
        S: Service<R, Error = BoxError>,
        S::Future: Send + 'static,
        S::Response: 'static,
    {
        type Response = S::Response;
        type Error = BoxError;
        type Future = Pin<Box<dyn Future<Output = Result<S::Response, BoxError>> + Send>>;

        fn poll_ready(&mut self, cx: &mut Context<'_>) -> Poll<Result<(), BoxError>> {
            self.0.poll_ready(cx)
        }

        fn call(&mut self, request: R) -> Self::Future {
            let delay = self.1;
            let connecting = self.0.call(request);
            Box::pin(async move {
                tokio::time::sleep(delay).await;
                connecting.await
            })
        }
    }

    async fn serve_after(head_delay: Duration) -> SocketAddr {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        tokio::spawn(async move {
            loop {
                let (mut socket, _) = listener.accept().await.unwrap();
                tokio::spawn(async move {
                    let mut buffer = [0_u8; 4096];
                    let _ = socket.read(&mut buffer).await;
                    tokio::time::sleep(head_delay).await;
                    let _ = socket
                        .write_all(b"HTTP/1.1 200 OK\r\ncontent-length: 2\r\n\r\nok")
                        .await;
                });
            }
        });
        address
    }

    fn client(connect_delay: Duration) -> Client {
        Client::new(
            reqwest::Client::builder()
                .no_proxy()
                .connector_layer(SlowConnect(connect_delay))
                .connector_layer(ConnectBound::new(Duration::from_secs(60)))
                .build()
                .unwrap(),
        )
    }

    fn request(address: SocketAddr) -> reqwest::Request {
        reqwest::Request::new(
            reqwest::Method::POST,
            format!("http://{address}/v1/messages").parse().unwrap(),
        )
    }

    enum Outcome {
        Sent,
        ConnectTimeout,
        ReadTimeout,
    }

    #[rstest]
    #[case::each_phase_within_its_own_limit(LIMIT * 3 / 4, LIMIT * 3 / 4, Outcome::Sent)]
    #[case::connect_over_its_limit(LIMIT * 2, Duration::ZERO, Outcome::ConnectTimeout)]
    #[case::head_over_the_read_limit(Duration::ZERO, LIMIT * 2, Outcome::ReadTimeout)]
    #[tokio::test]
    async fn each_phase_has_its_own_limit(
        #[case] connect_delay: Duration,
        #[case] head_delay: Duration,
        #[case] expected: Outcome,
    ) {
        let address = serve_after(head_delay).await;
        let phases = Phases {
            connect: LIMIT,
            read: LIMIT,
        };
        let result = client(connect_delay)
            .execute_in_phases(request(address), phases)
            .await;
        match expected {
            Outcome::Sent => assert_eq!(result.unwrap().status(), 200),
            Outcome::ConnectTimeout => {
                assert!(result.unwrap_err().is_connect_timeout());
            }
            Outcome::ReadTimeout => {
                let error = result.unwrap_err();
                assert!(error.is_timeout() && !error.is_connect_timeout());
            }
        }
    }

    #[tokio::test]
    async fn connect_limit_is_the_calls_own_not_the_clients() {
        let address = serve_after(Duration::ZERO).await;
        let client = Client::new(
            reqwest::Client::builder()
                .no_proxy()
                .connector_layer(SlowConnect(LIMIT * 2))
                .connector_layer(ConnectBound::new(LIMIT))
                .build()
                .unwrap(),
        );
        let phases = Phases {
            connect: LIMIT * 4,
            read: LIMIT,
        };
        let response = client.execute_in_phases(request(address), phases).await;
        assert_eq!(response.unwrap().status(), 200);
    }
}
