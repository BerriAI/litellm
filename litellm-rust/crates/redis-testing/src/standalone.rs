use testcontainers_modules::{
    redis::{REDIS_PORT, Redis},
    testcontainers::core::IntoContainerPort,
    testcontainers::{ContainerAsync, ImageExt, runners::AsyncRunner},
};

use crate::{
    Error,
    ports::{HOST, start_on_free_ports, wait_for_pong},
};

pub(crate) const REDIS_TAG: &str =
    "7.4@sha256:44960b3100c48159a66d28c18b0588c193492e25f7551d60e51dd464fc3eca5b";

pub struct RedisServer {
    container: ContainerAsync<Redis>,
    url: String,
}

impl RedisServer {
    pub async fn start() -> Result<Self, Error> {
        start_on_free_ports(Self::start_on).await
    }

    async fn start_on([host_port]: [u16; 1]) -> Result<Self, Error> {
        let container = Redis::default()
            .with_tag(REDIS_TAG)
            .with_mapped_port(host_port, REDIS_PORT.tcp())
            .start()
            .await?;
        wait_for_pong(host_port).await?;
        Ok(Self {
            container,
            url: format!("redis://{HOST}:{host_port}"),
        })
    }

    pub fn url(&self) -> &str {
        &self.url
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
