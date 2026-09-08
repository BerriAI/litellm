use std::sync::OnceLock;
use std::time::Duration;

use reqwest::Client;

use crate::constants::{HTTP_CONNECT_TIMEOUT_SECS, HTTP_TIMEOUT_SECS};

#[derive(Clone, Copy)]
pub(crate) enum HttpClientProfile {
    Standard,
    NoConnectTimeout,
    NoRedirectsOrDecompression,
}

pub(crate) fn http_client(
    profile: HttpClientProfile,
) -> Result<&'static Client, &'static reqwest::Error> {
    static STANDARD: OnceLock<Result<Client, reqwest::Error>> = OnceLock::new();
    static NO_CONNECT_TIMEOUT: OnceLock<Result<Client, reqwest::Error>> = OnceLock::new();
    static NO_REDIRECTS_OR_DECOMPRESSION: OnceLock<Result<Client, reqwest::Error>> =
        OnceLock::new();

    let slot = match profile {
        HttpClientProfile::Standard => &STANDARD,
        HttpClientProfile::NoConnectTimeout => &NO_CONNECT_TIMEOUT,
        HttpClientProfile::NoRedirectsOrDecompression => &NO_REDIRECTS_OR_DECOMPRESSION,
    };
    slot.get_or_init(|| build_client(profile)).as_ref()
}

fn build_client(profile: HttpClientProfile) -> Result<Client, reqwest::Error> {
    match profile {
        HttpClientProfile::Standard => Ok(Client::builder()
            .timeout(Duration::from_secs(HTTP_TIMEOUT_SECS))
            .connect_timeout(Duration::from_secs(HTTP_CONNECT_TIMEOUT_SECS))
            .build()
            .unwrap_or_else(|_| Client::new())),
        HttpClientProfile::NoConnectTimeout => Ok(Client::builder()
            .timeout(Duration::from_secs(HTTP_TIMEOUT_SECS))
            .build()
            .unwrap_or_else(|_| Client::new())),
        HttpClientProfile::NoRedirectsOrDecompression => Client::builder()
            .connect_timeout(Duration::from_secs(HTTP_CONNECT_TIMEOUT_SECS))
            .redirect(reqwest::redirect::Policy::none())
            .no_gzip()
            .no_brotli()
            .no_deflate()
            .no_zstd()
            .build(),
    }
}

#[cfg(test)]
mod tests {
    use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
    use tokio::net::TcpListener;

    use super::*;
    use crate::http_utils::http_request;

    #[tokio::test]
    async fn profiles_preserve_redirect_behavior() {
        for (profile, follows_redirects) in [
            (HttpClientProfile::Standard, true),
            (HttpClientProfile::NoConnectTimeout, true),
            (HttpClientProfile::NoRedirectsOrDecompression, false),
        ] {
            let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
            let url = format!("http://{}", listener.local_addr().unwrap());
            let redirect = format!(
                "HTTP/1.1 307 Temporary Redirect\r\nLocation: {url}/final\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
            );
            let server = tokio::spawn(async move {
                for index in 0..if follows_redirects { 2 } else { 1 } {
                    let (socket, _) = listener.accept().await.unwrap();
                    let mut socket = BufReader::new(socket);
                    let mut line = String::new();
                    socket.read_line(&mut line).await.unwrap();
                    assert_eq!(
                        line,
                        if index == 0 {
                            "GET / HTTP/1.1\r\n"
                        } else {
                            "GET /final HTTP/1.1\r\n"
                        }
                    );
                    loop {
                        line.clear();
                        assert_ne!(socket.read_line(&mut line).await.unwrap(), 0);
                        if line == "\r\n" {
                            break;
                        }
                    }
                    let response = if index == 0 {
                        redirect.as_bytes()
                    } else {
                        b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: close\r\n\r\nok"
                    };
                    socket.get_mut().write_all(response).await.unwrap();
                }
            });
            let response = http_request(
                http_client(profile)
                    .unwrap()
                    .get(url)
                    .timeout(Duration::from_secs(2)),
            )
            .await
            .unwrap();
            assert_eq!(
                response.status().as_u16(),
                if follows_redirects { 200 } else { 307 }
            );
            assert_eq!(
                response.text().await.unwrap(),
                if follows_redirects { "ok" } else { "" }
            );
            tokio::time::timeout(Duration::from_secs(2), server)
                .await
                .unwrap()
                .unwrap();
        }
    }

    #[tokio::test]
    async fn every_profile_honors_request_timeout_overrides() {
        for profile in [
            HttpClientProfile::Standard,
            HttpClientProfile::NoConnectTimeout,
            HttpClientProfile::NoRedirectsOrDecompression,
        ] {
            let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
            let url = format!("http://{}", listener.local_addr().unwrap());
            let server = tokio::spawn(async move {
                let (_socket, _) = listener.accept().await.unwrap();
                std::future::pending::<()>().await;
            });
            let result = tokio::time::timeout(
                Duration::from_secs(2),
                http_request(
                    http_client(profile)
                        .unwrap()
                        .get(url)
                        .timeout(Duration::from_millis(30)),
                ),
            )
            .await;
            server.abort();
            assert!(
                result
                    .expect("per-request timeout must override the client default")
                    .unwrap_err()
                    .is_timeout()
            );
        }
    }
}
