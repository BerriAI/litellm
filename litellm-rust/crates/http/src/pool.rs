use std::{
    collections::HashMap,
    sync::{Arc, Mutex, MutexGuard, PoisonError},
    time::{Duration, Instant},
};

use reqwest::dns::Resolve;

use crate::{client::Client, config::HttpClientConfig, error::Error, proxy::EnvironmentProxies};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ClientVariant {
    Provider,
    NoRedirect,
    Media,
    UnpinnedMedia,
}

const CLIENT_TTL: Duration = Duration::from_secs(3600);

struct PooledClient {
    client: reqwest::Client,
    built_at: Instant,
}

type Clients = HashMap<(HttpClientConfig, ClientVariant), PooledClient>;

pub struct HttpClientPool {
    media_resolver: Arc<dyn Resolve>,
    ttl: Duration,
    clients: Mutex<Clients>,
}

impl HttpClientPool {
    pub fn new(media_resolver: Arc<dyn Resolve>) -> Self {
        Self::with_ttl(media_resolver, CLIENT_TTL)
    }

    pub fn with_ttl(media_resolver: Arc<dyn Resolve>, ttl: Duration) -> Self {
        Self {
            media_resolver,
            ttl,
            clients: Mutex::default(),
        }
    }

    pub fn client(
        &self,
        config: &HttpClientConfig,
        variant: ClientVariant,
    ) -> Result<Client, Error> {
        let effective = match variant {
            ClientVariant::Media => HttpClientConfig {
                client_certificate: None,
                proxies: EnvironmentProxies::default(),
                ..config.clone()
            },
            ClientVariant::UnpinnedMedia => HttpClientConfig {
                client_certificate: None,
                ..config.clone()
            },
            ClientVariant::Provider | ClientVariant::NoRedirect => config.clone(),
        };
        let key = (effective, variant);
        if let Some(pooled) = self.lock().get(&key)
            && pooled.built_at.elapsed() < self.ttl
        {
            return Ok(Client::new(pooled.client.clone()));
        }
        let client = self
            .apply(variant, reqwest::ClientBuilder::try_from(&key.0)?)
            .build()?;
        self.lock().insert(
            key,
            PooledClient {
                client: client.clone(),
                built_at: Instant::now(),
            },
        );
        Ok(Client::new(client))
    }

    fn lock(&self) -> MutexGuard<'_, Clients> {
        self.clients.lock().unwrap_or_else(PoisonError::into_inner)
    }

    fn apply(
        &self,
        variant: ClientVariant,
        builder: reqwest::ClientBuilder,
    ) -> reqwest::ClientBuilder {
        match variant {
            ClientVariant::Provider => builder,
            ClientVariant::NoRedirect | ClientVariant::UnpinnedMedia => {
                builder.redirect(reqwest::redirect::Policy::none())
            }
            ClientVariant::Media => builder
                .redirect(reqwest::redirect::Policy::none())
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
    use crate::{ClientIdentity, HttpSettings, Resolution, Verify};

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
            ..Resolution::from(&HttpSettings::default()).config
        }
    }

    fn proxied_through(proxy: &str) -> EnvironmentProxies {
        let proxy = proxy.to_owned();
        EnvironmentProxies::from_environment(&move |name: &str| {
            (name == "HTTP_PROXY").then(|| proxy.clone())
        })
    }

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

    #[tokio::test]
    async fn provider_clients_route_through_the_resolved_proxy_not_the_process_environment() {
        let (proxy, connections, requests) = serve("HTTP/1.1 204 No Content").await;
        let config = HttpClientConfig {
            proxies: proxied_through(&format!("http://user:secret@{proxy}")),
            ..config("a")
        };
        let response = get(
            &pool(),
            &config,
            ClientVariant::Provider,
            "http://upstream.invalid/v1/ocr",
        )
        .await;
        assert_eq!(response.status(), 204);
        assert_eq!(connections.load(Ordering::SeqCst), 1);
        let request = requests.lock().unwrap().concat();
        assert!(request.starts_with("GET http://upstream.invalid/v1/ocr HTTP/1.1"));
        assert!(request.contains("proxy-authorization: Basic dXNlcjpzZWNyZXQ="));
    }

    #[tokio::test]
    async fn no_proxy_hosts_bypass_the_resolved_proxy() {
        let (upstream, _, _) = serve("HTTP/1.1 204 No Content").await;
        let (proxy, proxy_connections, _) = serve("HTTP/1.1 502 Bad Gateway").await;
        let config = HttpClientConfig {
            proxies: EnvironmentProxies::from_environment(&move |name: &str| match name {
                "HTTP_PROXY" => Some(format!("http://{proxy}")),
                "NO_PROXY" => Some("127.0.0.1".into()),
                _ => None,
            }),
            ..config("a")
        };
        let response = get(
            &pool(),
            &config,
            ClientVariant::Provider,
            &format!("http://{upstream}/v1/ocr"),
        )
        .await;
        assert_eq!(response.status(), 204);
        assert_eq!(proxy_connections.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn expired_clients_are_rebuilt() {
        let (address, connections, _) = serve("HTTP/1.1 204 No Content").await;
        let url = format!("http://{address}");
        let pool = HttpClientPool::with_ttl(
            Arc::new(FixedResolver(([192, 0, 2, 1], 80).into())),
            Duration::ZERO,
        );
        get(&pool, &config("a"), ClientVariant::Provider, &url).await;
        get(&pool, &config("a"), ClientVariant::Provider, &url).await;
        assert_eq!(connections.load(Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn media_clients_are_shared_across_proxy_settings_they_never_use() {
        let (address, connections, _) = serve("HTTP/1.1 204 No Content").await;
        let pool = HttpClientPool::new(Arc::new(FixedResolver(address)));
        let url = format!("http://media.invalid:{}/doc", address.port());
        for proxies in [
            proxied_through("http://proxy.invalid:3128"),
            EnvironmentProxies::default(),
        ] {
            let config = HttpClientConfig {
                proxies,
                ..config("a")
            };
            get(&pool, &config, ClientVariant::Media, &url).await;
        }
        assert_eq!(connections.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn media_variant_never_loads_the_client_certificate() {
        let pool = pool();
        let with_identity = HttpClientConfig {
            client_certificate: Some(ClientIdentity::Pem(
                std::env::temp_dir().join("litellm-http-absent-client.pem"),
            )),
            ..config("a")
        };
        assert!(
            pool.client(&with_identity, ClientVariant::Provider)
                .is_err()
        );
        assert!(pool.client(&with_identity, ClientVariant::Media).is_ok());
        assert!(
            pool.client(&with_identity, ClientVariant::UnpinnedMedia)
                .is_ok()
        );
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
    async fn unpinned_media_variant_uses_the_system_resolver_and_returns_redirects() {
        let (address, _, _) = serve("HTTP/1.1 302 Found").await;
        let pool = HttpClientPool::new(Arc::new(FixedResolver(([192, 0, 2, 1], 80).into())));
        let response = get(
            &pool,
            &config("a"),
            ClientVariant::UnpinnedMedia,
            &format!("http://localhost:{}/doc", address.port()),
        )
        .await;
        assert_eq!(response.status(), 302);
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
