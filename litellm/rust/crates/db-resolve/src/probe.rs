use std::time::Duration;

use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net::TcpStream,
    time::timeout,
};

const TIMEOUT: Duration = Duration::from_secs(5);
const SSL_REQUEST: [u8; 8] = [0, 0, 0, 8, 4, 210, 22, 47];
const REFUSED: u8 = b'N';

pub(crate) async fn server_offers_tls(host: &str, port: u16) -> bool {
    timeout(TIMEOUT, answer(host, port)).await.ok().flatten() != Some(REFUSED)
}

async fn answer(host: &str, port: u16) -> Option<u8> {
    let mut stream = TcpStream::connect((host, port)).await.ok()?;
    stream.write_all(&SSL_REQUEST).await.ok()?;
    stream.read_u8().await.ok()
}
