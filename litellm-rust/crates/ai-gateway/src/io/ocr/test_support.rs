use std::net::SocketAddr;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;
use tokio::task::JoinHandle;

pub(super) struct CountingServer {
    pub(super) addr: SocketAddr,
    connections: Arc<AtomicUsize>,
    handle: JoinHandle<()>,
}

impl CountingServer {
    pub(super) fn connection_count(&self) -> usize {
        self.connections.load(Ordering::SeqCst)
    }
}

impl Drop for CountingServer {
    fn drop(&mut self) {
        self.handle.abort();
    }
}

pub(super) async fn spawn_counting_server(response: Vec<u8>) -> CountingServer {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let connections = Arc::new(AtomicUsize::new(0));
    let counter = connections.clone();
    let handle = tokio::spawn(async move {
        loop {
            let Ok((mut socket, _)) = listener.accept().await else {
                return;
            };
            counter.fetch_add(1, Ordering::SeqCst);
            let mut discard = [0u8; 2048];
            let _ = socket.read(&mut discard).await;
            let _ = socket.write_all(&response).await;
            let _ = socket.flush().await;
        }
    });
    CountingServer {
        addr,
        connections,
        handle,
    }
}

pub(super) fn http_response(headers: &str, body: &[u8]) -> Vec<u8> {
    [headers.as_bytes(), body].concat()
}
