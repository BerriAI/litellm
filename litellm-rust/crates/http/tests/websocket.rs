use std::{
    net::SocketAddr,
    path::PathBuf,
    sync::{Arc, Mutex},
};

use futures_util::{SinkExt, StreamExt};
use http::{HeaderMap, HeaderValue};
use litellm_http::{
    ClientIdentity, ClientVariant, EnvironmentProxies, Error, HttpClientConfig, HttpClientPool,
    HttpSettings, Resolution, Verify, media::PublicDnsResolver, websocket::connect,
};
use rcgen::{BasicConstraints, CertificateParams, CertifiedIssuer, IsCa, KeyPair};
use rstest::rstest;
use rustls::{
    RootCertStore, ServerConfig,
    crypto::ring,
    pki_types::{CertificateDer, PrivateKeyDer},
    server::WebPkiClientVerifier,
};
use tokio::{
    io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt},
    net::{TcpListener, TcpStream},
};
use tokio_rustls::TlsAcceptor;
use tokio_tungstenite::{
    accept_hdr_async,
    tungstenite::{
        Message,
        handshake::server::{Request, Response},
    },
};

struct Pki {
    directory: tempfile::TempDir,
    ca: CertifiedIssuer<'static, KeyPair>,
    server_chain: Vec<CertificateDer<'static>>,
    server_key: Vec<u8>,
}

impl Pki {
    fn new() -> Self {
        let mut ca_params = CertificateParams::new(Vec::<String>::new()).unwrap();
        ca_params.is_ca = IsCa::Ca(BasicConstraints::Unconstrained);
        let ca = CertifiedIssuer::self_signed(ca_params, KeyPair::generate().unwrap()).unwrap();
        let server_key = KeyPair::generate().unwrap();
        let server = CertificateParams::new(vec!["localhost".into()])
            .unwrap()
            .signed_by(&server_key, &ca)
            .unwrap();
        let client_key = KeyPair::generate().unwrap();
        let client = CertificateParams::new(vec!["client".into()])
            .unwrap()
            .signed_by(&client_key, &ca)
            .unwrap();
        let directory = tempfile::tempdir().unwrap();
        std::fs::write(directory.path().join("ca.pem"), ca.pem()).unwrap();
        std::fs::write(
            directory.path().join("client.pem"),
            format!("{}{}", client.pem(), client_key.serialize_pem()),
        )
        .unwrap();
        Self {
            directory,
            server_chain: vec![server.der().clone()],
            server_key: server_key.serialize_der(),
            ca,
        }
    }

    fn path(&self, name: &str) -> PathBuf {
        self.directory.path().join(name)
    }

    fn acceptor(&self, require_client_certificate: bool) -> TlsAcceptor {
        let provider = Arc::new(ring::default_provider());
        let builder = ServerConfig::builder_with_provider(Arc::clone(&provider))
            .with_safe_default_protocol_versions()
            .unwrap();
        let builder = if require_client_certificate {
            let mut roots = RootCertStore::empty();
            roots.add(self.ca.der().clone()).unwrap();
            builder.with_client_cert_verifier(
                WebPkiClientVerifier::builder_with_provider(Arc::new(roots), provider)
                    .build()
                    .unwrap(),
            )
        } else {
            builder.with_no_client_auth()
        };
        let mut config = builder
            .with_single_cert(
                self.server_chain.clone(),
                PrivateKeyDer::try_from(self.server_key.clone()).unwrap(),
            )
            .unwrap();
        config.alpn_protocols = vec![b"h2".to_vec(), b"http/1.1".to_vec()];
        TlsAcceptor::from(Arc::new(config))
    }
}

fn config(verify: Verify) -> HttpClientConfig {
    HttpClientConfig {
        verify,
        http2: true,
        ..Resolution::from(&HttpSettings::default()).config
    }
}

fn proxied_through(proxy: SocketAddr) -> EnvironmentProxies {
    EnvironmentProxies::from_environment(&move |name: &str| {
        (name == "HTTPS_PROXY").then(|| format!("http://{proxy}"))
    })
}

async fn open(
    config: &HttpClientConfig,
    url: &str,
    headers: HeaderMap,
) -> Result<litellm_http::websocket::UpstreamWebSocket, Error> {
    let client = HttpClientPool::new(Arc::new(PublicDnsResolver))
        .client(config, ClientVariant::WebSocket)
        .unwrap();
    connect(&client, url, headers, None).await
}

async fn echo_once<S: AsyncRead + AsyncWrite + Unpin>(
    stream: S,
    seen: Arc<Mutex<Option<Request>>>,
) {
    #[expect(
        clippy::result_large_err,
        reason = "tungstenite requires an unboxed handshake error response"
    )]
    let Ok(mut socket) = accept_hdr_async(stream, |request: &Request, response: Response| {
        *seen.lock().unwrap() = Some(request.clone());
        Ok(response)
    })
    .await
    else {
        return;
    };
    if let Some(Ok(message)) = socket.next().await {
        let _ = socket.send(message).await;
    }
}

async fn serve_tls(
    pki: &Pki,
    require_client_certificate: bool,
) -> (SocketAddr, Arc<Mutex<Option<Request>>>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let acceptor = pki.acceptor(require_client_certificate);
    let seen = Arc::new(Mutex::new(None));
    let recorded = Arc::clone(&seen);
    tokio::spawn(async move {
        while let Ok((stream, _)) = listener.accept().await {
            if let Ok(stream) = acceptor.accept(stream).await {
                echo_once(stream, Arc::clone(&recorded)).await;
            }
        }
    });
    (address, seen)
}

async fn read_head(stream: &mut TcpStream) -> String {
    let mut head = Vec::new();
    while !head.ends_with(b"\r\n\r\n") {
        let mut byte = [0u8; 1];
        if stream.read(&mut byte).await.unwrap() == 0 {
            break;
        }
        head.push(byte[0]);
    }
    String::from_utf8(head).unwrap()
}

async fn serve_tunnel_proxy(target: SocketAddr) -> (SocketAddr, Arc<Mutex<Vec<String>>>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let requests = Arc::new(Mutex::new(Vec::new()));
    let seen = Arc::clone(&requests);
    tokio::spawn(async move {
        while let Ok((mut client, _)) = listener.accept().await {
            let head = read_head(&mut client).await;
            seen.lock()
                .unwrap()
                .push(head.lines().next().unwrap_or_default().to_owned());
            let mut upstream = TcpStream::connect(target).await.unwrap();
            client
                .write_all(b"HTTP/1.1 200 Connection established\r\n\r\n")
                .await
                .unwrap();
            let _ = tokio::io::copy_bidirectional(&mut client, &mut upstream).await;
        }
    });
    (address, requests)
}

async fn serve_raw(response: &'static str) -> SocketAddr {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    tokio::spawn(async move {
        let (mut stream, _) = listener.accept().await.unwrap();
        read_head(&mut stream).await;
        stream.write_all(response.as_bytes()).await.unwrap();
        let _ = stream.read(&mut [0u8; 1]).await;
    });
    address
}

async fn round_trip(socket: &mut litellm_http::websocket::UpstreamWebSocket) {
    socket.send(Message::Text("ping".into())).await.unwrap();
    assert_eq!(
        socket.next().await.unwrap().unwrap(),
        Message::Text("ping".into())
    );
}

#[rstest]
#[case::without_query("/responses")]
#[case::with_query("/responses?model=test-model")]
#[tokio::test]
async fn plain_websocket_sends_caller_headers_and_path(#[case] path: &'static str) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let seen = Arc::new(Mutex::new(None));
    let recorded = Arc::clone(&seen);
    tokio::spawn(async move {
        let (stream, _) = listener.accept().await.unwrap();
        echo_once(stream, recorded).await;
    });
    let headers = HeaderMap::from_iter([(
        http::header::AUTHORIZATION,
        HeaderValue::from_static("Bearer test-key"),
    )]);

    let mut socket = open(
        &config(Verify::BuiltInRoots),
        &format!("ws://{address}{path}"),
        headers,
    )
    .await
    .unwrap();

    round_trip(&mut socket).await;
    let request = seen.lock().unwrap().take().unwrap();
    assert_eq!(request.uri().path_and_query().unwrap().as_str(), path);
    assert_eq!(request.headers()["authorization"], "Bearer test-key");
}

#[rstest]
#[case::ca_bundle_trusts_the_private_ca(|pki: &Pki| Verify::CaBundle(pki.path("ca.pem")), true)]
#[case::ssl_verify_off_skips_verification(|_: &Pki| Verify::Disabled, true)]
#[case::built_in_roots_reject_the_private_ca(|_: &Pki| Verify::BuiltInRoots, false)]
#[tokio::test]
async fn wss_honors_ssl_verify(#[case] verify: fn(&Pki) -> Verify, #[case] trusted: bool) {
    let pki = Pki::new();
    let (address, _) = serve_tls(&pki, false).await;

    let result = open(
        &config(verify(&pki)),
        &format!("wss://localhost:{}/responses", address.port()),
        HeaderMap::new(),
    )
    .await;

    match result {
        Ok(mut socket) if trusted => round_trip(&mut socket).await,
        Err(error) if !trusted && !error.dispatched() => {}
        other => panic!("unexpected handshake outcome: {:?}", other.map(drop)),
    }
}

#[rstest]
#[case::with_certificate(true)]
#[case::without_certificate(false)]
#[tokio::test]
async fn wss_presents_the_configured_client_certificate(#[case] configured: bool) {
    let pki = Pki::new();
    let (address, _) = serve_tls(&pki, true).await;
    let config = HttpClientConfig {
        client_certificate: configured.then(|| ClientIdentity::Pem(pki.path("client.pem"))),
        ..config(Verify::CaBundle(pki.path("ca.pem")))
    };

    let result = open(
        &config,
        &format!("wss://localhost:{}/responses", address.port()),
        HeaderMap::new(),
    )
    .await;

    match result {
        Ok(mut socket) if configured => round_trip(&mut socket).await,
        Err(_) if !configured => {}
        other => panic!("unexpected handshake outcome: {:?}", other.map(drop)),
    }
}

#[rstest]
#[tokio::test]
async fn wss_tunnels_through_the_https_proxy() {
    let pki = Pki::new();
    let (target, _) = serve_tls(&pki, false).await;
    let (proxy, requests) = serve_tunnel_proxy(target).await;
    let config = HttpClientConfig {
        proxies: proxied_through(proxy),
        ..config(Verify::CaBundle(pki.path("ca.pem")))
    };

    let mut socket = open(
        &config,
        &format!("wss://localhost:{}/responses", target.port()),
        HeaderMap::new(),
    )
    .await
    .unwrap();

    round_trip(&mut socket).await;
    assert_eq!(
        *requests.lock().unwrap(),
        [format!("CONNECT localhost:{} HTTP/1.1", target.port())]
    );
}

#[rstest]
#[tokio::test]
async fn refused_upgrade_reports_the_status_and_body() {
    let address =
        serve_raw("HTTP/1.1 403 Forbidden\r\nContent-Length: 6\r\nConnection: close\r\n\r\ndenied")
            .await;

    let result = open(
        &config(Verify::BuiltInRoots),
        &format!("ws://{address}/responses"),
        HeaderMap::new(),
    )
    .await;

    let error = result.map(drop).unwrap_err();
    assert_eq!(error.upstream_response(), Some((403, "denied")));
}

#[rstest]
#[tokio::test]
async fn upgrade_with_a_wrong_accept_key_is_rejected() {
    let address = serve_raw(
        "HTTP/1.1 101 Switching Protocols\r\nConnection: Upgrade\r\nUpgrade: websocket\r\nSec-WebSocket-Accept: wrong\r\n\r\n",
    )
    .await;

    let result = open(
        &config(Verify::BuiltInRoots),
        &format!("ws://{address}/responses"),
        HeaderMap::new(),
    )
    .await;

    assert!(matches!(result, Err(error) if error.dispatched() && !error.is_request()));
}

#[rstest]
#[case::non_websocket_scheme("ftp://localhost/responses")]
#[case::unparseable("not a url")]
#[tokio::test]
async fn invalid_url_is_rejected_before_dispatch(#[case] url: &str) {
    let result = open(&config(Verify::BuiltInRoots), url, HeaderMap::new()).await;

    assert!(matches!(result, Err(error) if !error.dispatched() && error.is_request()));
}

#[rstest]
#[case::missing_upgrade(None, Some("Upgrade"), None, None, false)]
#[case::wrong_upgrade(Some("http"), Some("Upgrade"), None, None, false)]
#[case::missing_connection(Some("websocket"), None, None, None, false)]
#[case::wrong_connection(Some("websocket"), Some("keep-alive"), None, None, false)]
#[case::unsolicited_protocol(Some("websocket"), Some("Upgrade"), None, Some("other"), false)]
#[case::unoffered_protocol(
    Some("websocket"),
    Some("Upgrade"),
    Some("events"),
    Some("other"),
    false
)]
#[case::missing_protocol(Some("websocket"), Some("Upgrade"), Some("events"), None, false)]
#[case::matching_protocol(
    Some("websocket"),
    Some("Upgrade"),
    Some("events"),
    Some("events"),
    true
)]
#[case::multiple_offers(
    Some("websocket"),
    Some("Upgrade"),
    Some("events, realtime"),
    Some("realtime"),
    true
)]
#[case::case_insensitive_upgrade(Some("WebSocket"), Some("uPgRaDe"), None, None, true)]
#[case::connection_tokens(Some("websocket"), Some("keep-alive, Upgrade"), None, None, true)]
#[tokio::test]
async fn upgrade_validates_headers_and_negotiated_protocol(
    #[case] upgrade: Option<&'static str>,
    #[case] connection: Option<&'static str>,
    #[case] offered: Option<&'static str>,
    #[case] selected: Option<&'static str>,
    #[case] accepted: bool,
) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    tokio::spawn(async move {
        let (stream, _) = listener.accept().await.unwrap();
        #[expect(
            clippy::result_large_err,
            reason = "tungstenite requires an unboxed handshake error response"
        )]
        let result = accept_hdr_async(stream, move |_: &Request, mut response: Response| {
            response.headers_mut().remove("upgrade");
            response.headers_mut().remove("connection");
            if let Some(value) = upgrade {
                response
                    .headers_mut()
                    .insert("upgrade", HeaderValue::from_static(value));
            }
            if let Some(value) = connection {
                response
                    .headers_mut()
                    .insert("connection", HeaderValue::from_static(value));
            }
            if let Some(value) = selected {
                response
                    .headers_mut()
                    .insert("sec-websocket-protocol", HeaderValue::from_static(value));
            }
            Ok(response)
        })
        .await;
        let Ok(mut socket) = result else {
            return;
        };
        if let Some(Ok(message)) = socket.next().await {
            let _ = socket.send(message).await;
        }
    });
    let headers = offered.map_or_else(HeaderMap::new, |value| {
        HeaderMap::from_iter([(
            http::header::SEC_WEBSOCKET_PROTOCOL,
            HeaderValue::from_static(value),
        )])
    });
    let result = open(
        &config(Verify::BuiltInRoots),
        &format!("ws://{address}/responses"),
        headers,
    )
    .await;

    match result {
        Ok(mut socket) if accepted => round_trip(&mut socket).await,
        Err(error) if !accepted && error.dispatched() => {}
        other => panic!("unexpected handshake outcome: {:?}", other.map(drop)),
    }
}

#[rstest]
#[tokio::test]
async fn a_handshake_that_never_answers_times_out_as_dispatched() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    tokio::spawn(async move {
        let (mut stream, _) = listener.accept().await.unwrap();
        read_head(&mut stream).await;
        let _ = stream.read(&mut [0u8; 1]).await;
    });
    let client = HttpClientPool::new(Arc::new(PublicDnsResolver))
        .client(&config(Verify::BuiltInRoots), ClientVariant::WebSocket)
        .unwrap();

    let error = connect(
        &client,
        &format!("ws://{address}/responses"),
        HeaderMap::new(),
        Some(std::time::Duration::from_millis(100)),
    )
    .await
    .map(drop)
    .unwrap_err();

    assert!(error.dispatched());
}
