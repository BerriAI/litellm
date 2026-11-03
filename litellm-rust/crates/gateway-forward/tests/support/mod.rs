#![allow(
    dead_code,
    reason = "each test binary uses a different subset of the harness"
)]

use std::{future::Future, net::SocketAddr, time::Duration};

use axum::{
    Router,
    body::{Body, Bytes},
    extract::Request,
    http::{HeaderMap, Method, StatusCode, Uri},
};
use http_body_util::BodyExt;
use hyper_util::{
    client::legacy::{Client, connect::HttpConnector},
    rt::TokioExecutor,
};
use litellm_gateway_forward::Forward;
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net::{TcpListener, TcpStream},
    sync::{mpsc, oneshot},
    task::JoinHandle,
};

pub const WAIT: Duration = Duration::from_secs(5);
pub const UVICORN_KEEPALIVE: Duration = Duration::from_secs(5);

pub async fn within<T>(future: impl Future<Output = T>) -> T {
    tokio::time::timeout(WAIT, future).await.expect("timed out")
}

/// A running forwarder in front of `upstream`, listening on a real socket like production.
pub struct Gateway {
    pub address: SocketAddr,
    stop: oneshot::Sender<()>,
    server: JoinHandle<std::io::Result<()>>,
}

impl Gateway {
    pub async fn start(upstream: SocketAddr) -> Self {
        Self::with_keepalive(upstream, UVICORN_KEEPALIVE).await
    }

    pub async fn with_keepalive(upstream: SocketAddr, keepalive: Duration) -> Self {
        let forward = Forward::new(&format!("http://{upstream}"), keepalive).unwrap();
        let public = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = public.local_addr().unwrap();
        let (stop, stopped) = oneshot::channel::<()>();
        let server = tokio::spawn(forward.serve(public, async {
            stopped.await.ok();
        }));
        Self {
            address,
            stop,
            server,
        }
    }

    pub fn url(&self, path_and_query: &str) -> Uri {
        format!("http://{}{path_and_query}", self.address)
            .parse()
            .unwrap()
    }

    pub async fn connect(&self) -> Raw {
        Raw::new(TcpStream::connect(self.address).await.unwrap())
    }

    /// Starts graceful shutdown and returns the server task, which ends once in-flight work drains.
    pub fn shutdown(self) -> JoinHandle<std::io::Result<()>> {
        self.stop.send(()).unwrap();
        self.server
    }
}

pub fn client() -> Client<HttpConnector, Body> {
    Client::builder(TokioExecutor::new()).build_http()
}

/// What the upstream received, after the forwarder rewrote it.
pub struct Received {
    pub method: Method,
    pub uri: Uri,
    pub headers: HeaderMap,
    pub body: Bytes,
}

/// An axum upstream that records every request it fully reads and answers with `respond`.
pub async fn recording(
    respond: impl Fn() -> axum::response::Response + Clone + Send + Sync + 'static,
) -> (SocketAddr, mpsc::UnboundedReceiver<Received>) {
    let (sender, receiver) = mpsc::unbounded_channel();
    let address = serve(Router::new().fallback(move |request: Request| {
        let sender = sender.clone();
        let respond = respond.clone();
        async move {
            let (parts, body) = request.into_parts();
            let body = body.collect().await.unwrap().to_bytes();
            sender
                .send(Received {
                    method: parts.method,
                    uri: parts.uri,
                    headers: parts.headers,
                    body,
                })
                .unwrap();
            respond()
        }
    }))
    .await;
    (address, receiver)
}

pub async fn serve(router: Router) -> SocketAddr {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, router).await.unwrap() });
    address
}

/// An upstream that hands each accepted connection to `handle` as raw bytes, for behavior a
/// well-behaved HTTP server would never produce.
pub async fn raw_upstream<F, Fut>(handle: F) -> SocketAddr
where
    F: Fn(Raw) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = ()> + Send + 'static,
{
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    tokio::spawn(async move {
        loop {
            let (stream, _) = listener.accept().await.unwrap();
            tokio::spawn(handle(Raw::new(stream)));
        }
    });
    address
}

pub fn unreachable_upstream() -> SocketAddr {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    listener.local_addr().unwrap()
}

pub struct Head {
    pub status: StatusCode,
    pub headers: Vec<(String, Vec<u8>)>,
    pub path: String,
}

impl Head {
    pub fn header(&self, name: &str) -> Option<&[u8]> {
        self.headers
            .iter()
            .find(|(key, _)| key.eq_ignore_ascii_case(name))
            .map(|(_, value)| value.as_slice())
    }

    pub fn content_length(&self) -> Option<usize> {
        self.header("content-length")
            .map(|value| std::str::from_utf8(value).unwrap().parse().unwrap())
    }
}

/// One TCP connection driven byte by byte, so tests control framing, timing and disconnects.
/// Bytes read past a message head stay buffered for the next read.
pub struct Raw {
    stream: TcpStream,
    buffer: Vec<u8>,
}

impl Raw {
    pub fn new(stream: TcpStream) -> Self {
        Self {
            stream,
            buffer: Vec::new(),
        }
    }

    pub fn into_stream(self) -> TcpStream {
        assert!(self.buffer.is_empty(), "buffered bytes would be lost");
        self.stream
    }

    pub fn local_address(&self) -> SocketAddr {
        self.stream.local_addr().unwrap()
    }

    pub async fn send(&mut self, bytes: impl AsRef<[u8]>) {
        self.stream.write_all(bytes.as_ref()).await.unwrap();
    }

    /// Reads one request head, or `None` when the peer closed the connection first.
    pub async fn request_head(&mut self) -> Option<Head> {
        self.head(|bytes| {
            let mut headers = [httparse::EMPTY_HEADER; 256];
            let mut request = httparse::Request::new(&mut headers);
            let httparse::Status::Complete(length) = request.parse(bytes).unwrap() else {
                return None;
            };
            Some((
                length,
                Head {
                    status: StatusCode::OK,
                    headers: owned(request.headers),
                    path: request.path.unwrap().to_owned(),
                },
            ))
        })
        .await
    }

    pub async fn response_head(&mut self) -> Option<Head> {
        self.head(|bytes| {
            let mut headers = [httparse::EMPTY_HEADER; 256];
            let mut response = httparse::Response::new(&mut headers);
            let httparse::Status::Complete(length) = response.parse(bytes).unwrap() else {
                return None;
            };
            Some((
                length,
                Head {
                    status: StatusCode::from_u16(response.code.unwrap()).unwrap(),
                    headers: owned(response.headers),
                    path: String::new(),
                },
            ))
        })
        .await
    }

    async fn head(&mut self, parse: impl Fn(&[u8]) -> Option<(usize, Head)>) -> Option<Head> {
        loop {
            if let Some((length, head)) = parse(&self.buffer) {
                self.buffer.drain(..length);
                return Some(head);
            }
            if self.fill().await == 0 {
                return None;
            }
        }
    }

    /// Reads exactly `length` body bytes, using what is already buffered first.
    pub async fn body(&mut self, length: usize) -> Vec<u8> {
        while self.buffer.len() < length {
            assert_ne!(self.fill().await, 0, "closed mid-body");
        }
        self.buffer.drain(..length).collect()
    }

    /// Reads a whole chunked body and returns its decoded bytes.
    pub async fn chunked_body(&mut self) -> Vec<u8> {
        let mut decoded = Vec::new();
        loop {
            let size_line = self.line().await;
            let size =
                usize::from_str_radix(size_line.split(';').next().unwrap().trim(), 16).unwrap();
            if size == 0 {
                while !self.line().await.is_empty() {}
                return decoded;
            }
            decoded.extend(self.body(size).await);
            assert_eq!(self.body(2).await, b"\r\n");
        }
    }

    async fn line(&mut self) -> String {
        loop {
            if let Some(end) = self.buffer.windows(2).position(|window| window == b"\r\n") {
                let line = String::from_utf8(self.buffer[..end].to_vec()).unwrap();
                self.buffer.drain(..end + 2);
                return line;
            }
            assert_ne!(self.fill().await, 0, "closed mid-line");
        }
    }

    async fn fill(&mut self) -> usize {
        let mut chunk = [0; 8192];
        let read = self.stream.read(&mut chunk).await.unwrap_or(0);
        self.buffer.extend_from_slice(&chunk[..read]);
        read
    }

    /// Resolves once the peer closes the connection, discarding anything it still sends.
    pub async fn closed(&mut self) {
        let mut chunk = [0; 8192];
        while self.stream.read(&mut chunk).await.unwrap_or(0) != 0 {}
    }
}

fn owned(headers: &[httparse::Header<'_>]) -> Vec<(String, Vec<u8>)> {
    headers
        .iter()
        .map(|header| (header.name.to_owned(), header.value.to_vec()))
        .collect()
}
