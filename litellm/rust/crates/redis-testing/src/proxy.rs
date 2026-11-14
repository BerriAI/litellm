use std::sync::Arc;

use tokio::{
    net::{TcpListener, TcpStream},
    sync::watch,
    task::JoinHandle,
};

use crate::{Error, ports::HOST};

/// A TCP proxy in front of one Redis port, for outages a container cannot stage on one socket.
pub struct TcpProxy {
    port: u16,
    cuts: Arc<watch::Sender<u64>>,
    accepting: JoinHandle<()>,
}

impl TcpProxy {
    pub async fn start(upstream: u16) -> Result<Self, Error> {
        let listener = TcpListener::bind((HOST, 0)).await.map_err(Error::Port)?;
        let port = listener.local_addr().map_err(Error::Port)?.port();
        let cuts = Arc::new(watch::Sender::new(0));
        let accepting = tokio::spawn(accept(listener, upstream, cuts.clone()));
        Ok(Self {
            port,
            cuts,
            accepting,
        })
    }

    pub fn port(&self) -> u16 {
        self.port
    }

    /// Stops forwarding on every connection open now while keeping both its sockets open, the way
    /// a half-open TCP connection looks to the client: writes succeed and no reply ever comes.
    /// Connections opened afterwards forward normally.
    pub fn blackhole_open_connections(&self) {
        self.cuts.send_modify(|cut| *cut += 1);
    }
}

impl Drop for TcpProxy {
    fn drop(&mut self) {
        self.accepting.abort();
    }
}

async fn accept(listener: TcpListener, upstream: u16, cuts: Arc<watch::Sender<u64>>) {
    while let Ok((client, _)) = listener.accept().await {
        tokio::spawn(forward(client, upstream, cuts.subscribe()));
    }
}

async fn forward(mut client: TcpStream, upstream: u16, mut cut: watch::Receiver<u64>) {
    let Ok(mut server) = TcpStream::connect((HOST, upstream)).await else {
        return;
    };
    tokio::select! {
        _ = tokio::io::copy_bidirectional(&mut client, &mut server) => {}
        _ = cut.changed() => {
            while cut.changed().await.is_ok() {}
        }
    }
}
