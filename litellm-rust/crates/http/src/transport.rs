use bytes::{Bytes, BytesMut};

/// The provider never answered. A reply with any status is `http::Response<Bytes>`
/// from [`read`], so no variant here carries a status or headers.
#[derive(Clone, Debug, thiserror::Error, PartialEq, Eq)]
pub enum Error {
    #[error("upstream network error: {0}")]
    Network(String),
    #[error("could not reach the provider: {0}")]
    Connect(String),
    #[error("upstream request timed out: {0}")]
    Timeout(String),
    #[error("upstream response exceeds the size limit of {limit} bytes")]
    TooLarge { limit: usize },
}

impl Error {
    pub fn from_reqwest_before_dispatch(error: reqwest::Error) -> Self {
        let before_dispatch = !error.is_timeout() && (error.is_connect() || error.is_builder());
        if before_dispatch {
            Self::Connect(describe(error))
        } else {
            Self::from(error)
        }
    }
}

impl From<reqwest::Error> for Error {
    fn from(error: reqwest::Error) -> Self {
        if error.is_timeout() {
            Self::Timeout(describe(error))
        } else {
            Self::Network(describe(error))
        }
    }
}

fn describe(error: reqwest::Error) -> String {
    let error = error.without_url();
    std::iter::successors(std::error::Error::source(&error), |cause| cause.source())
        .fold(error.to_string(), |message, cause| {
            format!("{message}: {cause}")
        })
}

/// The status line and headers of a reply whose body is still streaming.
pub fn parts(response: &reqwest::Response) -> http::response::Parts {
    let (mut parts, ()) = http::Response::new(()).into_parts();
    parts.status = response.status();
    parts.version = response.version();
    parts.headers = response.headers().clone();
    parts
}

/// Drains the body and hands back the reply as the `http` types every other crate uses.
/// A successful reply over `limit` is [`Error::TooLarge`]; a failed reply is cut at `limit`
/// because its first bytes are what the caller reports.
pub async fn read(
    mut response: reqwest::Response,
    limit: Option<usize>,
) -> Result<http::Response<Bytes>, Error> {
    let limit = limit.unwrap_or(usize::MAX);
    let success = response.status().is_success();
    if success
        && response
            .content_length()
            .is_some_and(|length| length > limit as u64)
    {
        return Err(Error::TooLarge { limit });
    }
    let mut body = BytesMut::new();
    while let Some(chunk) = response.chunk().await? {
        let remaining = limit.saturating_sub(body.len());
        if success && chunk.len() > remaining {
            return Err(Error::TooLarge { limit });
        }
        body.extend_from_slice(&chunk[..chunk.len().min(remaining)]);
        if !success && body.len() == limit {
            break;
        }
    }
    let (parts, _) = http::Response::from(response).into_parts();
    Ok(http::Response::from_parts(parts, body.freeze()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn transport_errors_remove_urls_and_keep_dispatch_context() {
        let error = reqwest::Client::builder()
            .no_proxy()
            .build()
            .expect("client")
            .get("http://localhost:invalid/private?api_key=secret")
            .send()
            .await
            .expect_err("invalid port");
        let error = Error::from_reqwest_before_dispatch(error);
        assert!(matches!(error, Error::Connect(_)));
        assert!(!error.to_string().contains("secret"));
        assert!(!error.to_string().contains("private"));
    }

    fn root_cause(error: &dyn std::error::Error) -> Option<String> {
        match error.source() {
            Some(cause) => root_cause(cause).or_else(|| Some(cause.to_string())),
            None => None,
        }
    }

    #[tokio::test]
    async fn network_error_message_names_the_underlying_cause() {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind");
        let address = listener.local_addr().expect("address");
        drop(listener);
        let error = reqwest::Client::builder()
            .no_proxy()
            .build()
            .expect("client")
            .get(format!("http://{address}/private?api_key=secret"))
            .send()
            .await
            .expect_err("nothing listens on the port");
        let root_cause = root_cause(&error).expect("reqwest reports a cause");
        let message = Error::from(error).to_string();
        assert!(message.contains(&root_cause), "{message}");
        assert!(!message.contains("secret"));
    }

    #[tokio::test]
    async fn request_timeout_is_a_timeout_not_a_connect_failure() {
        use std::time::Duration;
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind");
        let address = listener.local_addr().expect("address");
        let request = reqwest::Client::builder()
            .no_proxy()
            .build()
            .expect("client")
            .get(format!("http://{address}"))
            .timeout(Duration::from_millis(200))
            .send();
        let (response, accepted) = tokio::join!(
            request,
            tokio::time::timeout(Duration::from_secs(2), listener.accept())
        );
        let _connection = accepted
            .expect("accept deadline")
            .expect("accepted connection");
        let error = response.expect_err("server does not respond");
        assert!(error.is_timeout());
        assert!(matches!(
            Error::from_reqwest_before_dispatch(error),
            Error::Timeout(_)
        ));
    }

    async fn serve_once(status: u16, body: &'static [u8]) -> String {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind");
        let address = listener.local_addr().expect("address");
        tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.expect("accept");
            let mut request = [0u8; 1024];
            let _ = socket.read(&mut request).await;
            let head = format!(
                "HTTP/1.1 {status} X\r\nx-provider-trace: first\r\nx-provider-trace: second\r\ncontent-length: {}\r\n\r\n",
                body.len()
            );
            socket.write_all(head.as_bytes()).await.expect("head");
            socket.write_all(body).await.expect("body");
        });
        format!("http://{address}")
    }

    #[rstest::rstest]
    #[case::success_within_limit(200, b"hello", Some(5), Ok(&b"hello"[..]))]
    #[case::success_unbounded(200, b"hello", None, Ok(&b"hello"[..]))]
    #[case::success_over_limit(200, b"hello", Some(4), Err(Error::TooLarge { limit: 4 }))]
    #[case::failure_is_cut_at_limit(503, b"hello", Some(4), Ok(&b"hell"[..]))]
    #[tokio::test]
    async fn read_keeps_status_duplicate_headers_and_bounds_the_body(
        #[case] status: u16,
        #[case] body: &'static [u8],
        #[case] limit: Option<usize>,
        #[case] expected: Result<&[u8], Error>,
    ) {
        let url = serve_once(status, body).await;
        let response = reqwest::Client::builder()
            .no_proxy()
            .build()
            .expect("client")
            .get(url)
            .send()
            .await
            .expect("reply");
        let result = read(response, limit).await;
        match expected {
            Ok(expected_body) => {
                let reply = result.expect("read");
                assert_eq!(reply.status().as_u16(), status);
                assert_eq!(
                    reply
                        .headers()
                        .get_all("x-provider-trace")
                        .iter()
                        .collect::<Vec<_>>(),
                    ["first", "second"]
                );
                assert_eq!(reply.body().as_ref(), expected_body);
            }
            Err(error) => assert_eq!(result.expect_err("too large"), error),
        }
    }
}
