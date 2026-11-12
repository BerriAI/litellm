use std::time::{Duration, Instant};

use testcontainers_modules::testcontainers::{
    ContainerAsync, GenericImage, ImageExt,
    core::{IntoContainerPort, WaitFor},
    runners::AsyncRunner,
};

use crate::{
    ClientConnection, Error,
    deployment::parse_client_list,
    ports::{HOST, start_on_free_ports, wait_for_pong},
    standalone::REDIS_TAG,
};

const READY: &str = "redis-testing cluster ready";
const ANNOUNCE_TIMEOUT: Duration = Duration::from_secs(30);

const START_CLUSTER: &str = r#"
set -e
bus=17000
for port in "$@"; do
  mkdir -p "/data/$port"
  redis-server --port "$port" --cluster-enabled yes --cluster-port "$bus" \
    --cluster-config-file "/data/$port/nodes.conf" --cluster-announce-ip 127.0.0.1 \
    --dir "/data/$port" --save '' --appendonly no --daemonize yes
  bus=$((bus + 1))
done
for port in "$@"; do
  until redis-cli -p "$port" ping >/dev/null 2>&1; do sleep 0.1; done
done
if [ ! -f /data/created ]; then
  redis-cli --cluster create $(for port in "$@"; do printf '127.0.0.1:%s ' "$port"; done) --cluster-yes >/dev/null
  touch /data/created
fi
for port in "$@"; do
  until redis-cli -p "$port" cluster info | grep -q cluster_state:ok; do sleep 0.1; done
done
echo "redis-testing cluster ready"
exec sleep infinity
"#;

pub struct RedisCluster {
    container: ContainerAsync<GenericImage>,
    ports: [u16; 3],
}

impl RedisCluster {
    pub async fn start() -> Result<Self, Error> {
        start_on_free_ports(Self::start_on).await
    }

    async fn start_on(ports: [u16; 3]) -> Result<Self, Error> {
        let image = GenericImage::new("redis", REDIS_TAG)
            .with_wait_for(WaitFor::message_on_stdout(READY))
            .with_entrypoint("sh");
        let request = ports.iter().fold(
            image.with_cmd(
                ["-c", START_CLUSTER, "sh"]
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

    pub fn nodes(&self) -> impl Iterator<Item = (&'static str, u16)> + '_ {
        self.ports.iter().map(|&port| (HOST, port))
    }

    /// One redis-cli reply per node, in `nodes()` order.
    pub async fn cli_outputs(&self, args: &[&str]) -> Result<Vec<String>, Error> {
        let mut replies = Vec::with_capacity(self.ports.len());
        for port in self.ports {
            replies.push(crate::exec::run(&self.container, crate::exec::cli(port, args)).await?);
        }
        Ok(replies)
    }

    pub async fn cli_on_every_node(&self, args: &[&str]) -> Result<(), Error> {
        for port in self.ports {
            crate::exec::run(&self.container, crate::exec::cli(port, args)).await?;
        }
        Ok(())
    }

    pub async fn cli_on_node(&self, port: u16, args: &[&str]) -> Result<String, Error> {
        crate::exec::run(&self.container, crate::exec::cli(port, args)).await
    }

    /// Makes each node tell clients it is at the port of `ports` in `nodes()` order, such as a
    /// `TcpProxy` in front of it, and returns once every node lists every announced port, so slot
    /// maps and redirects only lead a client to those ports.
    pub async fn announce_ports(&self, ports: [u16; 3]) -> Result<(), Error> {
        for (node, announced) in self.ports.into_iter().zip(ports) {
            self.cli_on_node(
                node,
                &[
                    "CONFIG",
                    "SET",
                    "cluster-announce-port",
                    &announced.to_string(),
                ],
            )
            .await?;
        }
        let deadline = Instant::now() + ANNOUNCE_TIMEOUT;
        loop {
            let views = self.cli_outputs(&["CLUSTER", "NODES"]).await?;
            let learned = views
                .iter()
                .all(|view| ports.iter().all(|port| view.contains(&format!(":{port}@"))));
            if learned {
                return Ok(());
            }
            if Instant::now() > deadline {
                return Err(Error::Announce);
            }
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
    }

    pub async fn clients(&self) -> Result<Vec<ClientConnection>, Error> {
        let mut clients = Vec::new();
        for port in self.ports {
            let reply =
                crate::exec::run(&self.container, crate::exec::cli(port, &["CLIENT", "LIST"]))
                    .await?;
            clients.extend(parse_client_list(port, &reply));
        }
        Ok(clients)
    }

    pub async fn restart(&self) -> Result<(), Error> {
        self.stop().await?;
        self.resume().await
    }

    pub async fn stop(&self) -> Result<(), Error> {
        Ok(self.container.stop().await?)
    }

    pub async fn resume(&self) -> Result<(), Error> {
        Ok(self.container.start().await?)
    }

    pub async fn freeze(&self) -> Result<(), Error> {
        Ok(self.container.pause().await?)
    }

    pub async fn thaw(&self) -> Result<(), Error> {
        Ok(self.container.unpause().await?)
    }
}
