use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use litellm_circuit_breaker::{Breaker, Config as BreakerConfig, WeakBreaker};
use litellm_redis_client::{RedisClient, RedisEvents, WeakRedisClient};
use litellm_redis_types::{Endpoint, Resolution};

use crate::{Error, HostServices, breaker, connect};

/// The Redis servers one host talks to. Every caller on a server shares its breaker, and callers with the same
/// settings share one client, which admits through its server's breaker. Both are held weakly, so each is freed
/// with its last user and the next caller builds a fresh one.
#[derive(Default)]
pub struct RedisHosts {
    breakers: Mutex<Vec<(Option<Endpoint>, WeakBreaker)>>,
    clients: Mutex<Vec<(ClientKey, WeakRedisClient)>>,
}

#[derive(PartialEq)]
struct ClientKey {
    resolved: Resolution,
    guarded: bool,
}

impl RedisHosts {
    /// The breaker of the server `endpoint` names, built from `config` by its first caller. Without an endpoint
    /// the breaker is this caller's own, which `breakers` still lists.
    pub fn breaker(
        &self,
        endpoint: Option<Endpoint>,
        config: BreakerConfig,
        events: Option<Arc<dyn RedisEvents>>,
    ) -> Breaker {
        let mut breakers = live(&self.breakers, WeakBreaker::upgrade);
        if let Some(shared) = breakers
            .iter()
            .filter(|(key, _)| key.is_some() && *key == endpoint)
            .find_map(|(_, weak)| weak.upgrade())
        {
            return shared;
        }
        let built = breaker(config, events);
        breakers.push((endpoint, built.downgrade()));
        built
    }

    /// The client for `resolved`, connected by its first caller and guarded by its server's breaker unless `breaker`
    /// is none. The lock is held while it connects, so concurrent callers with the same settings connect once.
    pub fn client(
        &self,
        resolved: Resolution,
        host: HostServices,
        breaker: Option<BreakerConfig>,
    ) -> Result<RedisClient, Error> {
        let breaker = breaker
            .map(|config| self.breaker(resolved.target.endpoint(), config, host.events.clone()));
        let key = ClientKey {
            resolved,
            guarded: breaker.is_some(),
        };
        let mut clients = live(&self.clients, WeakRedisClient::upgrade);
        if let Some(shared) = clients
            .iter()
            .filter(|(entry, _)| *entry == key)
            .find_map(|(_, weak)| weak.upgrade())
        {
            return Ok(shared);
        }
        let client = connect(key.resolved.clone(), host, breaker)?;
        clients.push((key, client.downgrade()));
        Ok(client)
    }

    /// Every breaker a caller still holds, for a state gauge.
    pub fn breakers(&self) -> Vec<Breaker> {
        live(&self.breakers, WeakBreaker::upgrade)
            .iter()
            .filter_map(|(_, weak)| weak.upgrade())
            .collect()
    }
}

fn live<K, W, V>(
    entries: &Mutex<Vec<(K, W)>>,
    upgrade: fn(&W) -> Option<V>,
) -> MutexGuard<'_, Vec<(K, W)>> {
    let mut entries = entries.lock().unwrap_or_else(PoisonError::into_inner);
    entries.retain(|(_, weak)| upgrade(weak).is_some());
    entries
}

#[cfg(test)]
mod tests {
    use std::{
        sync::Arc,
        time::{Duration, Instant},
    };

    use futures_util::FutureExt;
    use litellm_circuit_breaker::{Config, Failure, State};
    use litellm_redis_types::{
        Auth, Endpoint, HealthCheck, Resolution, Standalone, StaticAuth, Target,
    };
    use rstest::rstest;

    use super::RedisHosts;
    use crate::HostServices;

    const CONFIG: Config = Config {
        failure_threshold: 1,
        recovery_timeout: Duration::from_secs(60),
        timeout_min_duration: Duration::ZERO,
    };

    fn target(host: &str, db: i64) -> Target {
        Target::Standalone(Standalone {
            host: host.to_owned(),
            port: 6379,
            tls: None,
            db,
        })
    }

    fn endpoint(host: &str, db: i64) -> Option<Endpoint> {
        target(host, db).endpoint()
    }

    fn resolved(host: &str, namespace: Option<&str>) -> Resolution {
        Resolution {
            target: target(host, 0),
            auth: Auth::Static(StaticAuth::default()),
            namespace: namespace.map(str::to_owned),
            timeout: None,
            connect_timeout: None,
            health_check: HealthCheck::Unset,
            keepalive: true,
            ignored: Vec::new(),
        }
    }

    fn services() -> HostServices {
        HostServices {
            events: None,
            http: litellm_http::Client::for_test(),
            environment: Arc::new(|_: &str| None),
            credential_provider: None,
        }
    }

    fn open(breaker: &litellm_circuit_breaker::Breaker) {
        breaker.failed(Failure::Hard, Instant::now());
    }

    #[rstest]
    fn callers_on_one_server_share_its_breaker_whatever_their_db() {
        let hosts = RedisHosts::default();
        let first = hosts.breaker(endpoint("r1", 0), CONFIG, None);
        let second = hosts.breaker(endpoint("r1", 1), CONFIG, None);
        let other = hosts.breaker(endpoint("r2", 0), CONFIG, None);

        open(&first);

        assert!(matches!(second.state(), State::Open { .. }));
        assert!(matches!(other.state(), State::Closed { .. }));
    }

    #[rstest]
    fn a_breaker_without_an_endpoint_is_listed_but_never_shared() {
        let hosts = RedisHosts::default();
        let own = hosts.breaker(None, CONFIG, None);
        let shared = hosts.breaker(endpoint("r1", 0), CONFIG, None);

        open(&own);

        assert!(matches!(shared.state(), State::Closed { .. }));
        assert_eq!(hosts.breakers().len(), 2);
    }

    #[rstest]
    fn a_server_whose_callers_are_all_gone_gets_a_fresh_breaker() {
        let hosts = RedisHosts::default();
        let first = hosts.breaker(endpoint("r1", 0), CONFIG, None);
        open(&first);
        drop(first);

        let again = hosts.breaker(endpoint("r1", 0), CONFIG, None);

        assert!(matches!(again.state(), State::Closed { .. }));
        assert_eq!(hosts.breakers().len(), 1);
    }

    #[rstest]
    #[tokio::test]
    async fn callers_with_the_same_settings_share_one_client() {
        let hosts = RedisHosts::default();
        let first = hosts
            .client(resolved("r1", Some("a")), services(), Some(CONFIG))
            .unwrap();

        let same = hosts
            .client(resolved("r1", Some("a")), services(), Some(CONFIG))
            .unwrap();
        let other_namespace = hosts
            .client(resolved("r1", Some("b")), services(), Some(CONFIG))
            .unwrap();

        assert!(first.same_client(&same));
        assert!(!first.same_client(&other_namespace));
    }

    #[rstest]
    #[tokio::test]
    async fn a_client_admits_through_its_servers_breaker() {
        let hosts = RedisHosts::default();
        let breaker = hosts.breaker(endpoint("r1", 1), CONFIG, None);
        let client = hosts
            .client(resolved("r1", None), services(), Some(CONFIG))
            .unwrap();

        open(&breaker);

        assert!(matches!(
            client.ping().now_or_never(),
            Some(Err(litellm_redis_client::Error::BreakerOpen))
        ));
    }

    #[rstest]
    #[tokio::test]
    async fn a_client_without_a_breaker_is_never_shared_with_a_guarded_one() {
        let hosts = RedisHosts::default();
        let guarded = hosts
            .client(resolved("r1", None), services(), Some(CONFIG))
            .unwrap();

        let unguarded = hosts
            .client(resolved("r1", None), services(), None)
            .unwrap();

        assert!(!guarded.same_client(&unguarded));
    }
}
