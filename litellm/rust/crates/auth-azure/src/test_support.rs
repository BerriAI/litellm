use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use rcgen::{BasicConstraints, CertificateParams, CertifiedIssuer, IsCa, KeyPair};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};

pub(crate) const SERVED_TOKEN: &str = "requests-token";

/// A test CA whose bundle sits in a temporary file, and a token endpoint certificate it signed for
/// `127.0.0.1` and `login.microsoftonline.com`.
pub(crate) struct Pki {
    _directory: tempfile::TempDir,
    pub(crate) ca: PathBuf,
    server: Arc<rustls::ServerConfig>,
}

/// What a test token endpoint on `127.0.0.1` does with a connection.
#[derive(Clone, Copy, Debug)]
pub(crate) enum Endpoint {
    Token,
    Silent,
    ResetAfterRequest,
    CloseAfterRequest,
    ResetBeforeTls,
    CloseBeforeTls,
}

/// What a test HTTP proxy answers a `CONNECT` with.
#[derive(Clone, Copy, Debug)]
pub(crate) enum Proxy {
    Tunnel(u16),
    Status(&'static str),
    Close,
    Reset,
}

impl Pki {
    pub(crate) fn new() -> Self {
        Self::valid_until(time::OffsetDateTime::now_utc() + time::Duration::days(1))
    }

    pub(crate) fn expired() -> Self {
        Self::valid_until(time::OffsetDateTime::now_utc() - time::Duration::days(1))
    }

    fn valid_until(not_after: time::OffsetDateTime) -> Self {
        let mut ca_params = CertificateParams::new(Vec::<String>::new()).unwrap();
        ca_params.is_ca = IsCa::Ca(BasicConstraints::Unconstrained);
        let ca = CertifiedIssuer::self_signed(ca_params, KeyPair::generate().unwrap()).unwrap();
        let server_key = KeyPair::generate().unwrap();
        let mut server_params = CertificateParams::new(vec![
            "127.0.0.1".to_owned(),
            "login.microsoftonline.com".to_owned(),
        ])
        .unwrap();
        server_params.not_before = not_after - time::Duration::days(30);
        server_params.not_after = not_after;
        let server = server_params.signed_by(&server_key, &ca).unwrap();
        let directory = tempfile::tempdir().unwrap();
        let ca_path = directory.path().join("ca.pem");
        std::fs::write(&ca_path, ca.pem()).unwrap();
        let config = rustls::ServerConfig::builder_with_provider(Arc::new(
            rustls::crypto::ring::default_provider(),
        ))
        .with_safe_default_protocol_versions()
        .unwrap()
        .with_no_client_auth()
        .with_single_cert(
            vec![server.der().clone()],
            rustls::pki_types::PrivateKeyDer::try_from(server_key.serialize_der()).unwrap(),
        )
        .unwrap();
        Self {
            _directory: directory,
            ca: ca_path,
            server: Arc::new(config),
        }
    }

    pub(crate) async fn token_endpoint(&self) -> u16 {
        self.endpoint(Endpoint::Token).await
    }

    pub(crate) async fn endpoint(&self, behaviour: Endpoint) -> u16 {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        let acceptor = tokio_rustls::TlsAcceptor::from(Arc::clone(&self.server));
        tokio::spawn(async move {
            while let Ok((stream, _)) = listener.accept().await {
                tokio::spawn(serve(acceptor.clone(), stream, behaviour));
            }
        });
        port
    }
}

async fn serve(acceptor: tokio_rustls::TlsAcceptor, mut stream: TcpStream, behaviour: Endpoint) {
    match behaviour {
        Endpoint::ResetBeforeTls => {
            let _ = stream.read(&mut [0; 1]).await;
            reset(stream);
        }
        Endpoint::CloseBeforeTls => {
            let _ = stream.shutdown().await;
            std::future::pending::<()>().await;
        }
        Endpoint::Token
        | Endpoint::Silent
        | Endpoint::ResetAfterRequest
        | Endpoint::CloseAfterRequest => {
            let Ok(mut tls) = acceptor.accept(stream).await else {
                return;
            };
            match behaviour {
                Endpoint::Token => answer_token(&mut tls).await,
                Endpoint::Silent => std::future::pending().await,
                Endpoint::ResetAfterRequest => {
                    read_head(&mut tls).await;
                    reset(tls.into_inner().0);
                }
                _ => {
                    read_head(&mut tls).await;
                    let _ = tls.shutdown().await;
                }
            }
        }
    }
}

fn reset(stream: TcpStream) {
    let _ = stream.set_zero_linger();
    drop(stream);
}

async fn answer_token(stream: &mut (impl AsyncReadExt + AsyncWriteExt + Unpin)) {
    let head = read_head(stream).await;
    let length = head
        .lines()
        .find_map(|line| {
            let (name, value) = line.split_once(':')?;
            name.eq_ignore_ascii_case("content-length")
                .then(|| value.trim().parse::<usize>().ok())
                .flatten()
        })
        .unwrap_or(0);
    let mut body = vec![0; length];
    let _ = stream.read_exact(&mut body).await;
    let token = format!(
        r#"{{"token_type":"Bearer","expires_in":3600,"ext_expires_in":3600,"access_token":"{SERVED_TOKEN}"}}"#
    );
    let response = format!(
        "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{token}",
        token.len()
    );
    let _ = stream.write_all(response.as_bytes()).await;
    let _ = stream.shutdown().await;
}

async fn read_head(stream: &mut (impl AsyncReadExt + Unpin)) -> String {
    let mut head = Vec::new();
    let mut byte = [0; 1];
    while !head.ends_with(b"\r\n\r\n") {
        match stream.read(&mut byte).await {
            Ok(1) => head.push(byte[0]),
            _ => break,
        }
    }
    String::from_utf8_lossy(&head).into_owned()
}

/// An HTTP proxy that records each `CONNECT` target and answers it as `behaviour` says.
pub(crate) struct TunnelProxy {
    pub(crate) port: u16,
    targets: Arc<Mutex<Vec<String>>>,
}

impl TunnelProxy {
    pub(crate) async fn start(upstream: u16) -> Self {
        Self::answering(Proxy::Tunnel(upstream)).await
    }

    pub(crate) async fn answering(behaviour: Proxy) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        let targets = Arc::new(Mutex::new(Vec::new()));
        let recorded = Arc::clone(&targets);
        tokio::spawn(async move {
            while let Ok((mut client, _)) = listener.accept().await {
                let recorded = Arc::clone(&recorded);
                tokio::spawn(async move {
                    let head = read_head(&mut client).await;
                    let target = head
                        .split_whitespace()
                        .nth(1)
                        .unwrap_or_default()
                        .to_owned();
                    recorded.lock().unwrap().push(target);
                    match behaviour {
                        Proxy::Tunnel(upstream) => {
                            let Ok(mut server) = TcpStream::connect(("127.0.0.1", upstream)).await
                            else {
                                return;
                            };
                            let _ = client
                                .write_all(b"HTTP/1.1 200 Connection established\r\n\r\n")
                                .await;
                            let _ = tokio::io::copy_bidirectional(&mut client, &mut server).await;
                        }
                        Proxy::Status(status) => {
                            let _ = client
                                .write_all(
                                    format!("HTTP/1.1 {status}\r\ncontent-length: 0\r\n\r\n")
                                        .as_bytes(),
                                )
                                .await;
                        }
                        Proxy::Close => {}
                        Proxy::Reset => reset(client),
                    }
                });
            }
        });
        Self { port, targets }
    }

    pub(crate) fn targets(&self) -> Vec<String> {
        self.targets.lock().unwrap().clone()
    }
}
