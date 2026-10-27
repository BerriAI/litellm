use std::{
    future::Future,
    io,
    net::TcpListener,
    time::{Duration, Instant},
};

use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net::TcpStream,
};

use crate::Error;

pub(crate) const HOST: &str = "127.0.0.1";
const READY_TIMEOUT: Duration = Duration::from_secs(10);
const START_ATTEMPTS: usize = 3;

fn free_ports<const N: usize>() -> Result<[u16; N], Error> {
    let listeners = (0..N)
        .map(|_| TcpListener::bind((HOST, 0)))
        .collect::<io::Result<Vec<_>>>()
        .map_err(Error::Port)?;
    let ports = listeners
        .iter()
        .map(|listener| listener.local_addr().map(|address| address.port()))
        .collect::<io::Result<Vec<_>>>()
        .map_err(Error::Port)?;
    Ok(ports
        .try_into()
        .unwrap_or_else(|ports: Vec<u16>| unreachable!("{} ports", ports.len())))
}

pub(crate) async fn wait_for_pong(port: u16) -> Result<(), Error> {
    let deadline = Instant::now() + READY_TIMEOUT;
    while Instant::now() < deadline {
        if ping(port).await.unwrap_or(false) {
            return Ok(());
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    Err(Error::Unreachable { port })
}

async fn ping(port: u16) -> io::Result<bool> {
    let mut stream = TcpStream::connect((HOST, port)).await?;
    stream.write_all(b"PING\r\n").await?;
    let mut reply = [0; 7];
    stream.read_exact(&mut reply).await?;
    Ok(&reply == b"+PONG\r\n")
}

pub(crate) async fn start_on_free_ports<const N: usize, T, F>(
    start: impl Fn([u16; N]) -> F,
) -> Result<T, Error>
where
    F: Future<Output = Result<T, Error>>,
{
    let mut attempt = 1;
    loop {
        match start(free_ports()?).await {
            Err(_) if attempt < START_ATTEMPTS => attempt += 1,
            started => return started,
        }
    }
}
