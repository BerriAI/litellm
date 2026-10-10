use std::{
    collections::HashMap,
    net::{IpAddr, Ipv4Addr},
    sync::{Mutex, PoisonError},
    time::{Duration, Instant},
};

use crate::{Error, HttpClientConfig};

struct Entry {
    client: reqwest_mcp::Client,
    built_at: Instant,
}

#[derive(Default)]
pub(super) struct Pool(Mutex<HashMap<HttpClientConfig, Entry>>);

impl Pool {
    pub fn client(
        &self,
        config: &HttpClientConfig,
        ttl: Duration,
    ) -> Result<reqwest_mcp::Client, Error> {
        let mut clients = self.0.lock().unwrap_or_else(PoisonError::into_inner);
        if let Some(entry) = clients.get(config)
            && entry.built_at.elapsed() < ttl
        {
            return Ok(entry.client.clone());
        }
        let client = build(config)?;
        clients.insert(
            config.clone(),
            Entry {
                client: client.clone(),
                built_at: Instant::now(),
            },
        );
        Ok(client)
    }
}

fn build(config: &HttpClientConfig) -> Result<reqwest_mcp::Client, Error> {
    let base = reqwest_mcp::Client::builder()
        .tls_backend_preconfigured(rustls::ClientConfig::try_from(config)?)
        .connect_timeout(config.connect_timeout)
        .pool_idle_timeout(config.pool_idle_timeout)
        .redirect(reqwest_mcp::redirect::Policy::none());
    let keepalive = match config.tcp_keepalive {
        Some(value) => base
            .tcp_keepalive(value.idle)
            .tcp_keepalive_interval(value.interval)
            .tcp_keepalive_retries(value.retries),
        None => base,
    };
    let address = if config.force_ipv4 {
        keepalive.local_address(IpAddr::V4(Ipv4Addr::UNSPECIFIED))
    } else {
        keepalive
    };
    let protocol = if config.http2 {
        address
    } else {
        address.http1_only()
    };
    let agent = match &config.user_agent {
        Some(value) => protocol.user_agent(value),
        None => protocol,
    };
    config
        .proxies
        .mcp_proxies()
        .into_iter()
        .fold(agent.no_proxy(), reqwest_mcp::ClientBuilder::proxy)
        .build()
        .map_err(|error| Error::Client(error.without_url().to_string()))
}
