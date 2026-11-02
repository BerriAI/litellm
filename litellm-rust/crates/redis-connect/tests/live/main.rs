mod aws;
mod azure;
mod gcp;

use std::{path::PathBuf, sync::Arc, time::Duration};

use litellm_config::{RedisSettings, SecretSnapshot};
use litellm_http::{
    ClientVariant, HttpClientPool, HttpSettings, HttpSettingsLayer, Resolution,
    media::PublicDnsResolver,
};
use litellm_redis_connect::{HostServices, connect};
use serde::Deserialize;

#[derive(Deserialize)]
struct LiveConfig {
    redis: RedisSettings,
    hold_seconds: u64,
}

fn config(variable: &str) -> Option<LiveConfig> {
    let path = PathBuf::from(std::env::var_os(variable)?);
    Some(
        serde_json::from_slice(&std::fs::read(path).expect("read live Redis config"))
            .expect("parse live Redis config"),
    )
}

async fn roundtrip(config: LiveConfig) {
    if rustls::crypto::CryptoProvider::get_default().is_none() {
        let _ = rustls::crypto::ring::default_provider().install_default();
    }
    let lookup = |name: &str| std::env::var(name).ok();
    let resolved = config
        .redis
        .resolve(
            &SecretSnapshot::read(config.redis.secret_names(), &lookup),
            &lookup,
        )
        .expect("resolve live Redis config");
    let http_settings = HttpSettings::from_layers([HttpSettingsLayer::from_environment(&lookup)]);
    let pool = HttpClientPool::new(Arc::new(PublicDnsResolver));
    let http = pool
        .client(
            &Resolution::from(&http_settings).config,
            ClientVariant::Provider,
        )
        .expect("build production HTTP client");
    let services = || HostServices {
        events: None,
        breaker: None,
        http: http.clone(),
        environment: Arc::new(|name: &str| std::env::var(name).ok()),
    };
    let client = connect(resolved.clone(), services()).expect("build live Redis client");
    let marker = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let key = format!("litellm:redis-connect:live:{}:{marker}", std::process::id());
    let ttl = Duration::from_secs(config.hold_seconds + 120);

    client.ping().await.expect("authenticate and ping");
    client
        .set(&key, "before-refresh", ttl)
        .await
        .expect("set before refresh");
    assert_eq!(
        client.get::<String>(&key).await.unwrap().as_deref(),
        Some("before-refresh")
    );

    tokio::time::sleep(Duration::from_secs(config.hold_seconds)).await;

    assert_eq!(
        client.get::<String>(&key).await.unwrap().as_deref(),
        Some("before-refresh")
    );
    let fresh = connect(resolved, services()).expect("build a fresh live Redis client");
    assert_eq!(
        fresh.get::<String>(&key).await.unwrap().as_deref(),
        Some("before-refresh")
    );
    fresh
        .set(&key, "after-refresh", ttl)
        .await
        .expect("set after refresh");
    assert_eq!(
        client.get::<String>(&key).await.unwrap().as_deref(),
        Some("after-refresh")
    );
    client.del(&[&key]).await.expect("delete live test key");
}
