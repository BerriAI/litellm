use std::time::{Duration, Instant};

use testcontainers_modules::testcontainers::{
    ContainerAsync, GenericImage, ImageExt,
    core::{IntoContainerPort, WaitFor},
    runners::AsyncRunner,
};

use crate::{
    ClientConnection, Error,
    deployment::parse_client_list,
    exec::{cli, run},
    ports::{HOST, start_on_free_ports, wait_for_pong},
    standalone::REDIS_TAG,
};

pub const SERVICE_NAME: &str = "primary";
const READY: &str = "redis-testing sentinel ready";
const FAILOVER_TIMEOUT: Duration = Duration::from_secs(30);

const START_SENTINEL: &str = r#"
set -e
primary=$1 replica=$2 sentinel=$3
common="--save '' --appendonly no --daemonize yes --replica-announce-ip 127.0.0.1"
eval redis-server --port "$primary" $common
eval redis-server --port "$replica" --replicaof 127.0.0.1 "$primary" $common
until redis-cli -p "$replica" info replication | grep -q master_link_status:up; do sleep 0.1; done
cat > /data/sentinel.conf <<CONF
port $sentinel
sentinel announce-ip 127.0.0.1
sentinel monitor primary 127.0.0.1 $primary 1
sentinel down-after-milliseconds primary 1000
sentinel failover-timeout primary 5000
CONF
redis-server /data/sentinel.conf --sentinel --daemonize yes
until redis-cli -p "$sentinel" sentinel replicas primary | grep -q "$replica"; do sleep 0.1; done
echo "redis-testing sentinel ready"
exec sleep infinity
"#;

pub struct RedisSentinel {
    container: ContainerAsync<GenericImage>,
    ports: [u16; 3],
}

impl RedisSentinel {
    pub async fn start() -> Result<Self, Error> {
        start_on_free_ports(Self::start_on).await
    }

    async fn start_on(ports: [u16; 3]) -> Result<Self, Error> {
        let image = GenericImage::new("redis", REDIS_TAG)
            .with_wait_for(WaitFor::message_on_stdout(READY))
            .with_entrypoint("sh");
        let request = ports.iter().fold(
            image.with_cmd(
                ["-c", START_SENTINEL, "sh"]
                    .into_iter()
                    .map(str::to_owned)
                    .chain(ports.iter().map(u16::to_string)),
            ),
            |request, &port| request.with_mapped_port(port, port.tcp()),
        );
        let container = request.start().await?;
        for port in ports {
            wait_for_pong(port).await?;
        }
        Ok(Self { container, ports })
    }

    pub fn sentinel(&self) -> (&'static str, u16) {
        (HOST, self.ports[2])
    }

    /// Runs one redis-cli command on the primary and the replica, which keep their own ACLs.
    pub async fn cli_on_data_nodes(&self, args: &[&str]) -> Result<(), Error> {
        for port in [self.ports[0], self.ports[1]] {
            run(&self.container, cli(port, args)).await?;
        }
        Ok(())
    }

    pub async fn clients_on(&self, port: u16) -> Result<Vec<ClientConnection>, Error> {
        Ok(parse_client_list(
            port,
            &run(&self.container, cli(port, &["CLIENT", "LIST"])).await?,
        ))
    }

    pub fn primary_port(&self) -> u16 {
        self.ports[0]
    }

    pub fn replica_port(&self) -> u16 {
        self.ports[1]
    }

    pub async fn master_port(&self) -> Result<u16, Error> {
        let reply = run(
            &self.container,
            cli(
                self.ports[2],
                &["sentinel", "get-master-addr-by-name", SERVICE_NAME],
            ),
        )
        .await?;
        Ok(reply
            .lines()
            .nth(1)
            .and_then(|port| port.trim().parse().ok())
            .unwrap_or_default())
    }

    pub async fn failover(&self) -> Result<u16, Error> {
        let before = self.master_port().await?;
        run(
            &self.container,
            cli(self.ports[2], &["sentinel", "failover", SERVICE_NAME]),
        )
        .await?;
        let deadline = Instant::now() + FAILOVER_TIMEOUT;
        loop {
            let now = self.master_port().await?;
            let promoted = now != before && self.role(now).await?.starts_with("master");
            if promoted && self.role(before).await?.starts_with("slave") {
                return Ok(now);
            }
            if Instant::now() > deadline {
                return Err(Error::Failover);
            }
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
    }

    async fn role(&self, port: u16) -> Result<String, Error> {
        run(&self.container, cli(port, &["role"])).await
    }
}
