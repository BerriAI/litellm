use rstest::{fixture, rstest};

use super::{LiveConfig, config, roundtrip};

#[fixture]
fn live_config() -> Option<LiveConfig> {
    config("LITELLM_REDIS_AZURE_LIVE_CONFIG")
}

#[rstest]
#[tokio::test]
async fn credentials_authenticate_existing_and_fresh_connections(live_config: Option<LiveConfig>) {
    let Some(config) = live_config else {
        return;
    };
    roundtrip(config).await;
}
