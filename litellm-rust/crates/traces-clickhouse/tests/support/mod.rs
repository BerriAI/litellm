use litellm_http::Client;
use rstest::fixture;
use testcontainers_modules::{
    clickhouse::ClickHouse,
    testcontainers::{ContainerAsync, ImageExt, runners::AsyncRunner},
};

const CLICKHOUSE_TAG: &str =
    "26.9.6.6@sha256:eb4870e7ca7ed70c259eebfcfbee6cf797017f6b5436c2926bbbfe3d4d28486e";

pub type TestResult<T = ()> = Result<T, Box<dyn std::error::Error>>;

pub struct ClickHouseDatabase {
    _container: ContainerAsync<ClickHouse>,
    pub url: String,
    pub client: Client,
}

#[fixture]
pub async fn database() -> TestResult<ClickHouseDatabase> {
    let container = ClickHouse::default()
        .with_tag(CLICKHOUSE_TAG)
        .with_env_var("CLICKHOUSE_SKIP_USER_SETUP", "1")
        .start()
        .await?;
    let url = format!(
        "http://{}:{}",
        container.get_host().await?,
        container.get_host_port_ipv4(8123).await?
    );
    Ok(ClickHouseDatabase {
        _container: container,
        url,
        client: Client::no_redirect_for_test(),
    })
}
