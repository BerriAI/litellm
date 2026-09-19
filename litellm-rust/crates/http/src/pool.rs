use std::{
    collections::HashMap,
    sync::{Arc, Mutex, PoisonError},
};

use reqwest::dns::Resolve;

use crate::{config::HttpClientConfig, error::Error};

/// The client shapes routes need; each is the shared base plus one policy.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ClientVariant {
    Provider,
    NoRedirect,
    /// Media downloads: no redirects (the fetcher validates each hop), never a proxy, and the
    /// pool's media resolver.
    Media,
}

/// Counterpart of `get_async_httpx_client`: one `reqwest::Client` per resolved configuration
/// and variant, built on first use and shared afterwards.
pub struct HttpClientPool {
    media_resolver: Arc<dyn Resolve>,
    clients: Mutex<HashMap<(HttpClientConfig, ClientVariant), reqwest::Client>>,
}

impl HttpClientPool {
    pub fn new(media_resolver: Arc<dyn Resolve>) -> Self {
        Self {
            media_resolver,
            clients: Mutex::default(),
        }
    }

    pub fn client(
        &self,
        config: &HttpClientConfig,
        variant: ClientVariant,
    ) -> Result<reqwest::Client, Error> {
        let key = (config.clone(), variant);
        let mut clients = self.clients.lock().unwrap_or_else(PoisonError::into_inner);
        if let Some(client) = clients.get(&key) {
            return Ok(client.clone());
        }
        let client = self.apply(variant, config.client_builder()?).build()?;
        clients.insert(key, client.clone());
        Ok(client)
    }

    fn apply(
        &self,
        variant: ClientVariant,
        builder: reqwest::ClientBuilder,
    ) -> reqwest::ClientBuilder {
        match variant {
            ClientVariant::Provider => builder,
            ClientVariant::NoRedirect => builder.redirect(reqwest::redirect::Policy::none()),
            ClientVariant::Media => builder
                .redirect(reqwest::redirect::Policy::none())
                .no_proxy()
                .dns_resolver2(Arc::clone(&self.media_resolver)),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::{
        net::SocketAddr,
        sync::atomic::{AtomicUsize, Ordering},
        time::Duration,
    };

    use reqwest::dns::{Addrs, Name, Resolving};
    use tokio::{
        io::{AsyncReadExt, AsyncWriteExt},
        net::TcpListener,
    };

    use super::*;
    use crate::{HttpSettings, Verify};

    struct FixedResolver(SocketAddr);

    impl Resolve for FixedResolver {
        fn resolve(&self, _: Name) -> Resolving {
            let addrs: Addrs = Box::new(std::iter::once(self.0));
            Box::pin(std::future::ready(Ok(addrs)))
        }
    }

    fn pool() -> HttpClientPool {
        HttpClientPool::new(Arc::new(FixedResolver(([192, 0, 2, 1], 80).into())))
    }

    fn config(user_agent: &str) -> HttpClientConfig {
        HttpClientConfig {
            user_agent: Some(user_agent.into()),
            ..HttpClientConfig::resolve(&HttpSettings::default()).unwrap()
        }
    }

    /// Answers every request on every connection with `status_line` and counts connections,
    /// so a reused client shows up as a reused keep-alive connection.
    async fn serve(
        status_line: &'static str,
    ) -> (SocketAddr, Arc<AtomicUsize>, Arc<Mutex<Vec<String>>>) {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let connections = Arc::new(AtomicUsize::new(0));
        let requests = Arc::new(Mutex::new(Vec::new()));
        let (accepted, seen) = (Arc::clone(&connections), Arc::clone(&requests));
        tokio::spawn(async move {
            loop {
                let (mut socket, _) = listener.accept().await.unwrap();
                accepted.fetch_add(1, Ordering::SeqCst);
                let seen = Arc::clone(&seen);
                tokio::spawn(async move {
                    let mut buffer = vec![0u8; 4096];
                    while let Ok(read) = socket.read(&mut buffer).await {
                        if read == 0 {
                            return;
                        }
                        seen.lock()
                            .unwrap()
                            .push(String::from_utf8_lossy(&buffer[..read]).into_owned());
                        let response = format!(
                            "{status_line}\r\nLocation: /elsewhere\r\nContent-Length: 0\r\n\r\n"
                        );
                        if socket.write_all(response.as_bytes()).await.is_err() {
                            return;
                        }
                    }
                });
            }
        });
        (address, connections, requests)
    }

    async fn get(
        pool: &HttpClientPool,
        config: &HttpClientConfig,
        variant: ClientVariant,
        url: &str,
    ) -> reqwest::Response {
        pool.client(config, variant)
            .unwrap()
            .get(url)
            .timeout(Duration::from_secs(5))
            .send()
            .await
            .unwrap()
    }

    #[tokio::test]
    async fn clients_are_shared_per_config_and_variant() {
        let (address, connections, _) = serve("HTTP/1.1 204 No Content").await;
        let url = format!("http://{address}");
        let pool = pool();
        get(&pool, &config("a"), ClientVariant::Provider, &url).await;
        get(&pool, &config("a"), ClientVariant::Provider, &url).await;
        assert_eq!(connections.load(Ordering::SeqCst), 1);
        get(&pool, &config("a"), ClientVariant::NoRedirect, &url).await;
        assert_eq!(connections.load(Ordering::SeqCst), 2);
        get(&pool, &config("b"), ClientVariant::Provider, &url).await;
        assert_eq!(connections.load(Ordering::SeqCst), 3);
    }

    #[test]
    fn build_failures_are_not_cached() {
        let pool = pool();
        let missing = HttpClientConfig {
            verify: Verify::CaBundle(std::env::temp_dir().join("litellm-http-absent.pem")),
            ..config("a")
        };
        assert!(pool.client(&missing, ClientVariant::Provider).is_err());
        assert!(pool.client(&missing, ClientVariant::Provider).is_err());
        assert!(pool.client(&config("a"), ClientVariant::Provider).is_ok());
    }

    #[tokio::test]
    async fn provider_client_sends_the_configured_user_agent_over_http1() {
        let (address, _, requests) = serve("HTTP/1.1 204 No Content").await;
        let response = get(
            &pool(),
            &config("litellm-test/9"),
            ClientVariant::Provider,
            &format!("http://{address}"),
        )
        .await;
        assert_eq!(response.status(), 204);
        assert_eq!(response.version(), reqwest::Version::HTTP_11);
        let request = requests.lock().unwrap()[0].clone();
        assert!(request.contains("user-agent: litellm-test/9"), "{request}");
    }

    #[tokio::test]
    async fn no_redirect_variant_returns_the_redirect_instead_of_following_it() {
        let (address, _, _) = serve("HTTP/1.1 302 Found").await;
        let response = get(
            &pool(),
            &config("a"),
            ClientVariant::NoRedirect,
            &format!("http://{address}"),
        )
        .await;
        assert_eq!(response.status(), 302);
        assert_eq!(response.headers()["location"], "/elsewhere");
    }

    #[tokio::test]
    async fn media_variant_resolves_through_the_injected_resolver() {
        let (address, _, requests) = serve("HTTP/1.1 204 No Content").await;
        let pool = HttpClientPool::new(Arc::new(FixedResolver(address)));
        let url = format!("http://media.invalid:{}/doc", address.port());
        let response = get(&pool, &config("a"), ClientVariant::Media, &url).await;
        assert_eq!(response.status(), 204);
        assert!(requests.lock().unwrap()[0].contains("host: media.invalid"));
        assert!(
            pool.client(&config("a"), ClientVariant::Provider)
                .unwrap()
                .get(&url)
                .timeout(Duration::from_secs(5))
                .send()
                .await
                .is_err()
        );
    }
}
