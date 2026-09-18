use std::{
    collections::HashMap,
    sync::{Mutex, PoisonError},
};

use crate::config::{Error, HttpClientConfig};

/// The client shapes routes need; each is the shared base plus one policy.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ClientVariant {
    Provider,
    NoRedirect,
    /// Media downloads: no redirects (the fetcher validates each hop) and never a proxy.
    Media,
}

impl ClientVariant {
    fn apply(self, builder: reqwest::ClientBuilder) -> reqwest::ClientBuilder {
        match self {
            Self::Provider => builder,
            Self::NoRedirect => builder.redirect(reqwest::redirect::Policy::none()),
            Self::Media => builder
                .redirect(reqwest::redirect::Policy::none())
                .no_proxy(),
        }
    }
}

/// Counterpart of `get_async_httpx_client`: one `reqwest::Client` per resolved configuration
/// and variant, built on first use and shared afterwards.
#[derive(Default)]
pub struct HttpClientPool {
    clients: Mutex<HashMap<(HttpClientConfig, ClientVariant), reqwest::Client>>,
}

impl HttpClientPool {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn client(
        &self,
        config: &HttpClientConfig,
        variant: ClientVariant,
    ) -> Result<reqwest::Client, Error> {
        self.client_with(config, variant, |builder| builder)
    }

    /// Like [`Self::client`], with a caller hook for builder options that are not plain values
    /// (a DNS resolver, for example). The hook only runs when the client is first built.
    pub fn client_with(
        &self,
        config: &HttpClientConfig,
        variant: ClientVariant,
        customize: impl FnOnce(reqwest::ClientBuilder) -> reqwest::ClientBuilder,
    ) -> Result<reqwest::Client, Error> {
        let key = (config.clone(), variant);
        let mut clients = self.clients.lock().unwrap_or_else(PoisonError::into_inner);
        if let Some(client) = clients.get(&key) {
            return Ok(client.clone());
        }
        let client = customize(variant.apply(config.client_builder()?)).build()?;
        clients.insert(key, client.clone());
        Ok(client)
    }
}

#[cfg(test)]
mod tests {
    use std::{cell::Cell, time::Duration};

    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    use super::*;
    use crate::{HttpSettings, Verify};

    fn config(user_agent: &str) -> HttpClientConfig {
        HttpClientConfig {
            user_agent: Some(user_agent.into()),
            ..HttpClientConfig::resolve(&HttpSettings::default(), None).unwrap()
        }
    }

    #[test]
    fn clients_are_built_once_per_config_and_variant() {
        let pool = HttpClientPool::new();
        let builds = Cell::new(0);
        let build = |config: &HttpClientConfig, variant| {
            pool.client_with(config, variant, |builder| {
                builds.set(builds.get() + 1);
                builder
            })
            .unwrap()
        };
        build(&config("a"), ClientVariant::Provider);
        build(&config("a"), ClientVariant::Provider);
        assert_eq!(builds.get(), 1);
        build(&config("a"), ClientVariant::NoRedirect);
        assert_eq!(builds.get(), 2);
        build(&config("b"), ClientVariant::Provider);
        assert_eq!(builds.get(), 3);
        build(&config("b"), ClientVariant::Provider);
        build(&config("a"), ClientVariant::NoRedirect);
        assert_eq!(builds.get(), 3);
    }

    #[test]
    fn build_failures_are_not_cached() {
        let pool = HttpClientPool::new();
        let missing = HttpClientConfig {
            verify: Verify::CaBundle(std::env::temp_dir().join("litellm-http-absent.pem")),
            ..config("a")
        };
        assert!(pool.client(&missing, ClientVariant::Provider).is_err());
        assert!(pool.client(&missing, ClientVariant::Provider).is_err());
        assert!(pool.client(&config("a"), ClientVariant::Provider).is_ok());
    }

    async fn serve_once(status_line: &'static str) -> (String, tokio::task::JoinHandle<String>) {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let base = format!("http://{}", listener.local_addr().unwrap());
        let server = tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut request = vec![0u8; 4096];
            let read = socket.read(&mut request).await.unwrap();
            socket
                .write_all(
                    format!("{status_line}\r\nLocation: /elsewhere\r\nContent-Length: 0\r\nConnection: close\r\n\r\n")
                        .as_bytes(),
                )
                .await
                .unwrap();
            String::from_utf8_lossy(&request[..read]).into_owned()
        });
        (base, server)
    }

    #[tokio::test]
    async fn provider_client_sends_the_configured_user_agent_over_http1() {
        let (base, server) = serve_once("HTTP/1.1 204 No Content").await;
        let config = HttpClientConfig {
            connect_timeout: Duration::from_secs(2),
            ..config("litellm-test/9")
        };
        let response = HttpClientPool::new()
            .client(&config, ClientVariant::Provider)
            .unwrap()
            .get(&base)
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), 204);
        assert_eq!(response.version(), reqwest::Version::HTTP_11);
        let request = server.await.unwrap();
        assert!(request.contains("user-agent: litellm-test/9"), "{request}");
    }

    #[tokio::test]
    async fn no_redirect_variant_returns_the_redirect_instead_of_following_it() {
        let (base, server) = serve_once("HTTP/1.1 302 Found").await;
        let response = HttpClientPool::new()
            .client(&config("a"), ClientVariant::NoRedirect)
            .unwrap()
            .get(&base)
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), 302);
        assert_eq!(response.headers()["location"], "/elsewhere");
        server.await.unwrap();
    }
}
