#![allow(dead_code)] // each test binary compiles this module on its own and uses a different subset

use std::{
    sync::{Arc, Mutex},
    time::Duration,
};

use litellm_circuit_breaker::{State, Transition};
use litellm_redis_client::{
    CallEvent, Cluster, Credentials, Namespace, Node, Operation, RedisClient, RedisConfig,
    RedisEvents, RedisUrl, Target, Timeouts,
};
use litellm_redis_testing::RedisDeployment;

pub fn connect(deployment: &RedisDeployment) -> RedisClient {
    connect_with(deployment, None, None)
}

/// Every round trip a client made, as the client reports it.
#[derive(Default)]
pub struct RoundTrips(Mutex<Vec<Operation>>);

impl RoundTrips {
    pub fn seen(&self) -> Vec<Operation> {
        self.0.lock().unwrap().clone()
    }
}

impl RedisEvents for RoundTrips {
    fn call(&self, event: &CallEvent) {
        self.0.lock().unwrap().push(event.operation);
    }

    fn breaker(&self, _transition: &Transition<State>) {}
}

pub fn connect_observed(deployment: &RedisDeployment) -> (RedisClient, Arc<RoundTrips>) {
    let trips = Arc::new(RoundTrips::default());
    (connect_with(deployment, None, Some(trips.clone())), trips)
}

pub fn connect_guarded(
    deployment: &RedisDeployment,
    breaker: litellm_circuit_breaker::Breaker,
) -> RedisClient {
    connect_with(deployment, Some(breaker), None)
}

fn connect_with(
    deployment: &RedisDeployment,
    breaker: Option<litellm_circuit_breaker::Breaker>,
    events: Option<Arc<dyn RedisEvents>>,
) -> RedisClient {
    let endpoint = match deployment {
        RedisDeployment::Standalone(server) => Target::Url(RedisUrl {
            url: litellm_secret_value::SecretValue::new(server.url().to_owned()),
            tls: None,
            db: None,
            username: None,
            password: None,
        }),
        RedisDeployment::Cluster(cluster) => Target::Cluster(Cluster {
            nodes: cluster
                .nodes()
                .map(|(host, port)| Node {
                    host: host.to_owned(),
                    port,
                })
                .collect(),
            tls: None,
        }),
    };
    RedisClient::new(RedisConfig {
        target: endpoint,
        credentials: Credentials::NONE,
        namespace: Namespace::default(),
        timeouts: Timeouts::new(Duration::from_millis(500), None),
        breaker,
        events,
        health_check: None,
    })
    .unwrap()
}
