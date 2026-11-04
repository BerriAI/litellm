use testcontainers_modules::testcontainers::{
    ContainerAsync, GenericImage, ImageExt,
    core::{IntoContainerPort, WaitFor},
    runners::AsyncRunner,
};

use crate::{
    Error,
    ports::{HOST, start_on_free_ports, wait_for_pong},
    standalone::REDIS_TAG,
};

const READY: &str = "redis-testing cluster ready";

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
