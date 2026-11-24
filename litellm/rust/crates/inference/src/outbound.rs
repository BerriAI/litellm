use std::time::Duration;

use bytes::Bytes;

use litellm_http::{
    Phases,
    outbound::{json, json_bytes, signed},
};
use litellm_llms::base_llm::auth::Authenticated;
use serde_json::Value;

#[tracing::instrument(
    name = "litellm.provider.send",
    level = "debug",
    skip_all,
    fields(status)
)]
pub async fn send(
    request: reqwest::Request,
    client: &litellm_http::Client,
) -> Result<reqwest::Response, litellm_http::Error> {
    let response = client.execute(request).await?;
    tracing::Span::current().record("status", response.status().as_u16());
    Ok(response)
}

/// Sends `request` with httpx's phase limits. With `retry_connect`, a connection that fails to
/// open is tried once more, as Python's async client retries it through a fresh client.
#[tracing::instrument(
    name = "litellm.provider.send",
    level = "debug",
    skip_all,
    fields(status)
)]
pub async fn send_in_phases(
    request: reqwest::Request,
    client: &litellm_http::Client,
    phases: Phases,
    retry_connect: bool,
) -> Result<reqwest::Response, litellm_http::Error> {
    let retry = retry_connect.then(|| request.try_clone()).flatten();
    let response = match (client.execute_in_phases(request, phases).await, retry) {
        (Err(error), Some(retry)) if error.is_connect() => {
            client.execute_in_phases(retry, phases).await?
        }
        (response, _) => response?,
    };
    tracing::Span::current().record("status", response.status().as_u16());
    Ok(response)
}

/// Header credentials are already in `headers`; SigV4 is applied here, over the
/// bytes that are sent.
pub fn outbound_request(
    authenticated: Authenticated,
    url: &str,
    body: &Value,
    timeout: Option<Duration>,
) -> Result<reqwest::Request, litellm_http::Error> {
    let Authenticated { headers, signer } = authenticated;
    let request = json(url, &headers, body, timeout)?;
    match signer {
        None => Ok(request),
        Some(signer) => signed(request, &signer),
    }
}

pub fn raw_leaf_body(body: &Value) -> Result<Bytes, litellm_http::Error> {
    litellm_host::raw_json::to_vec(body)
        .map(Bytes::from)
        .map_err(litellm_http::Error::request_body)
}

pub fn serialized_outbound_request(
    authenticated: Authenticated,
    url: &str,
    body: Bytes,
    timeout: Option<Duration>,
) -> Result<reqwest::Request, litellm_http::Error> {
    let Authenticated { headers, signer } = authenticated;
    let request = json_bytes(url, &headers, body, timeout)?;
    match signer {
        None => Ok(request),
        Some(signer) => signed(request, &signer),
    }
}

#[cfg(test)]
mod tests {
    use std::{
        net::SocketAddr,
        sync::{
            Arc,
            atomic::{AtomicUsize, Ordering},
        },
    };

    use reqwest::dns::{Addrs, Name, Resolve, Resolving};
    use rstest::rstest;
    use tokio::{
        io::{AsyncReadExt, AsyncWriteExt},
        net::TcpListener,
    };

    use super::*;

    struct UnresolvedOnce {
        address: SocketAddr,
        lookups: Arc<AtomicUsize>,
    }

    impl Resolve for UnresolvedOnce {
        fn resolve(&self, _: Name) -> Resolving {
            let first = self.lookups.fetch_add(1, Ordering::SeqCst) == 0;
            let address = self.address;
            Box::pin(async move {
                if first {
                    return Err(std::io::Error::other("Name or service not known").into());
                }
                let addrs: Addrs = Box::new(std::iter::once(address));
                Ok(addrs)
            })
        }
    }

    async fn serve() -> SocketAddr {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        tokio::spawn(async move {
            loop {
                let (mut socket, _) = listener.accept().await.unwrap();
                tokio::spawn(async move {
                    let mut buffer = [0_u8; 4096];
                    let _ = socket.read(&mut buffer).await;
                    let _ = socket
                        .write_all(b"HTTP/1.1 200 OK\r\ncontent-length: 2\r\n\r\nok")
                        .await;
                });
            }
        });
        address
    }

    #[rstest]
    #[case::retried(true, 2)]
    #[case::not_retried(false, 1)]
    #[tokio::test]
    async fn a_connection_that_fails_to_open_is_retried_once(
        #[case] retry_connect: bool,
        #[case] expected_lookups: usize,
    ) {
        let address = serve().await;
        let lookups = Arc::new(AtomicUsize::new(0));
        #[expect(
            clippy::disallowed_methods,
            clippy::disallowed_types,
            reason = "the pool has no resolver setting to fail the first lookup"
        )]
        let client = litellm_http::Client::wrap_for_test(
            reqwest::Client::builder()
                .dns_resolver(Arc::new(UnresolvedOnce {
                    address,
                    lookups: Arc::clone(&lookups),
                }))
                .no_proxy()
                .build()
                .unwrap(),
        );
        let request = json(
            &format!("http://provider.test:{}/v1/messages", address.port()),
            &[],
            &serde_json::json!({}),
            None,
        )
        .unwrap();
        let phases = Phases {
            connect: Duration::from_secs(5),
            read: Duration::from_secs(5),
        };
        let result = send_in_phases(request, &client, phases, retry_connect).await;
        assert_eq!(lookups.load(Ordering::SeqCst), expected_lookups);
        match result {
            Ok(response) => assert!(retry_connect && response.status() == 200),
            Err(error) => assert!(!retry_connect && error.is_connect()),
        }
    }
}
