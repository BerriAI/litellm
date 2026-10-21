use futures_util::{SinkExt, StreamExt};
use litellm_http::websocket::connect_upstream;
use rstest::rstest;
use tokio::net::TcpListener;
use tokio_tungstenite::{
    accept_hdr_async,
    tungstenite::{
        Message,
        client::IntoClientRequest,
        handshake::server::{Request, Response},
    },
};

#[rstest]
#[case::without_query("/responses")]
#[case::with_query("/responses?model=test-model")]
#[tokio::test]
async fn connects_with_caller_headers_and_exchanges_frames(#[case] path: &'static str) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let server = tokio::spawn(async move {
        let (stream, _) = listener.accept().await.unwrap();
        #[expect(
            clippy::result_large_err,
            reason = "tungstenite requires an unboxed handshake error response"
        )]
        let mut socket = accept_hdr_async(stream, move |request: &Request, response: Response| {
            assert_eq!(request.uri().path_and_query().unwrap().as_str(), path);
            assert_eq!(request.headers()["authorization"], "Bearer test-key");
            Ok(response)
        })
        .await
        .unwrap();
        assert_eq!(
            socket.next().await.unwrap().unwrap(),
            Message::Text("request".into())
        );
        socket.send(Message::Text("response".into())).await.unwrap();
        socket.close(None).await.unwrap();
    });
    let mut request = format!("ws://{address}{path}")
        .into_client_request()
        .unwrap();
    request
        .headers_mut()
        .insert("authorization", "Bearer test-key".parse().unwrap());
    let (mut socket, response) = connect_upstream(request).await.unwrap();
    assert_eq!(response.status(), 101);
    socket.send(Message::Text("request".into())).await.unwrap();
    assert_eq!(
        socket.next().await.unwrap().unwrap(),
        Message::Text("response".into())
    );
    assert!(matches!(
        socket.next().await.unwrap().unwrap(),
        Message::Close(_)
    ));
    server.await.unwrap();
}
