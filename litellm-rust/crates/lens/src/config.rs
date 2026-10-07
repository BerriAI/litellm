use crate::Error;
use litellm_http::{
    Client, ClientVariant, HttpClientPool, HttpSettings, Resolution, media::PublicDnsResolver,
};
use litellm_traces_clickhouse::Config as StorageConfig;
use std::{net::SocketAddr, sync::Arc, time::Duration};

pub struct Config {
    pub address: SocketAddr,
    pub proxy_url: url::Url,
    pub worker_token: String,
    pub service_token: String,
    pub release: String,
    pub storage: StorageConfig,
}

fn required(name: &'static str) -> Result<String, Error> {
    std::env::var(name)
        .ok()
        .filter(|value| !value.is_empty())
        .ok_or(Error::Configuration(name))
}

impl Config {
    pub fn from_env() -> Result<Self, Error> {
        let proxy_url = url::Url::parse(&required("LITELLM_URL")?)
            .map_err(|_| Error::Configuration("LITELLM_URL"))?;
        if !matches!(proxy_url.scheme(), "http" | "https")
            || !proxy_url.username().is_empty()
            || proxy_url.password().is_some()
            || proxy_url.query().is_some()
            || proxy_url.fragment().is_some()
        {
            return Err(Error::Configuration("LITELLM_URL"));
        }
        let worker_token = required("LENS_WORKER_TOKEN")?;
        let service_token = required("LITELLM_LENS_SERVICE_TOKEN")?;
        if service_token.len() < 32 {
            return Err(Error::Configuration(
                "LITELLM_LENS_SERVICE_TOKEN must contain at least 32 characters",
            ));
        }
        Ok(Self {
            address: std::env::var("LITELLM_LENS_LISTEN")
                .unwrap_or_else(|_| "0.0.0.0:4318".into())
                .parse()
                .map_err(|_| Error::Configuration("LITELLM_LENS_LISTEN"))?,
            proxy_url,
            worker_token,
            service_token,
            release: required("LITELLM_RELEASE_TAG")?,
            storage: StorageConfig::new(
                std::env::var("CLICKHOUSE_DATABASE").unwrap_or_else(|_| "litellm".into()),
                &required("CLICKHOUSE_URL")?,
                std::env::var("AGENT_TRACING_RETENTION_DAYS")
                    .unwrap_or_else(|_| "14".into())
                    .parse()
                    .map_err(|_| Error::Configuration("AGENT_TRACING_RETENTION_DAYS"))?,
                65_536,
            )?,
        })
    }
}

pub fn http_client() -> Result<Client, Error> {
    let settings = HttpSettings {
        connect_timeout: Duration::from_secs(5),
        ..HttpSettings::default()
    };
    Ok(HttpClientPool::new(Arc::new(PublicDnsResolver)).client(
        &Resolution::from(&settings).config,
        ClientVariant::NoRedirect,
    )?)
}
