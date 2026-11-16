mod support;

use std::{fmt, net::SocketAddr};

use axum::{
    Router,
    body::{Body, Bytes},
    extract::Request,
    http::Method,
    response::Response,
    routing::post,
};
use http_body_util::BodyExt;
use litellm_gateway_forward::{Forward, VERDICT_HEADER};
use rstest::rstest;
use support::{DRAIN, Received, UVICORN_KEEPALIVE, client, recording, within};
use tokio::{net::TcpListener, sync::mpsc};

const FOR_PYTHON: &str = "x-for-python";

struct Marked;

impl fmt::Display for Marked {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("the client marked the request for python")
    }
}

impl From<&Marked> for &'static str {
    fn from(_: &Marked) -> Self {
        "marked"
    }
}

async fn admit(request: Request) -> Result<Request, (Box<Request>, Marked)> {
    match request.headers().contains_key(FOR_PYTHON) {
        true => Err((Box::new(request), Marked)),
        false => Ok(request),
    }
}

fn native() -> Router {
    Router::new().route("/native", post(|body: Bytes| async move { body }))
}

async fn python() -> (SocketAddr, mpsc::UnboundedReceiver<Received>) {
    recording(|| Response::new(Body::from("python"))).await
}

async fn start(upstream: SocketAddr, app: impl FnOnce(Forward) -> Router) -> SocketAddr {
    let forward = Forward::new(
        &format!("http://{upstream}").parse().unwrap(),
        UVICORN_KEEPALIVE,
    );
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let app = app(forward.clone());
    tokio::spawn(forward.serve_app(app, listener, std::future::pending(), DRAIN));
    address
}

struct Answer {
    verdict: String,
    body: Bytes,
}

async fn send(gateway: SocketAddr, method: Method, path: &str, marked: bool) -> Answer {
    let request = Request::builder()
        .method(method)
        .uri(format!("http://{gateway}{path}"));
    let request = match marked {
        true => request.header(FOR_PYTHON, "1"),
        false => request,
    };
    let response = within(client().request(request.body(Body::from("payload")).unwrap()))
        .await
        .unwrap();
    Answer {
        verdict: response.headers()[VERDICT_HEADER]
            .to_str()
            .unwrap()
            .to_owned(),
        body: within(response.into_body().collect())
            .await
            .unwrap()
            .to_bytes(),
    }
}

#[rstest]
#[tokio::test]
async fn an_admitted_request_is_answered_natively_and_never_reaches_python() {
    let (upstream, mut received) = python().await;
    let gateway = start(upstream, |python| python.behind(native(), admit)).await;

    let answer = send(gateway, Method::POST, "/native", false).await;

    assert_eq!(answer.verdict, "native");
    assert_eq!(answer.body, "payload");
    assert!(received.try_recv().is_err());
}

#[rstest]
#[case::refused(Method::POST, "/native", true, "python; reason=marked")]
#[case::no_route(Method::POST, "/elsewhere", false, "python; reason=route")]
#[case::wrong_method(Method::PUT, "/native", false, "python; reason=route")]
#[tokio::test]
async fn python_answers_what_the_native_app_does_not_and_gets_it_as_sent(
    #[case] method: Method,
    #[case] path: &str,
    #[case] marked: bool,
    #[case] verdict: &str,
) {
    let (upstream, mut received) = python().await;
    let gateway = start(upstream, |python| python.behind(native(), admit)).await;

    let answer = send(gateway, method.clone(), path, marked).await;

    assert_eq!(answer.verdict, verdict);
    assert_eq!(answer.body, "python");
    let forwarded = within(received.recv()).await.unwrap();
    assert_eq!(forwarded.method, method);
    assert_eq!(forwarded.uri.path(), path);
    assert_eq!(forwarded.body, "payload");
    assert_eq!(forwarded.headers.contains_key(FOR_PYTHON), marked);
}

#[rstest]
#[tokio::test]
async fn without_a_native_app_python_answers_every_request() {
    let (upstream, mut received) = python().await;
    let gateway = start(upstream, Forward::without_native).await;

    let answer = send(gateway, Method::POST, "/native", false).await;

    assert_eq!(answer.verdict, "python; reason=configuration");
    assert_eq!(answer.body, "python");
    assert_eq!(within(received.recv()).await.unwrap().body, "payload");
}
