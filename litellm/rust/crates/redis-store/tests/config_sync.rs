mod support;

use std::time::Duration;

use futures_util::{StreamExt, stream::BoxStream};
use litellm_redis_store::config_sync::{self, ConfigChange, Heard, channel};
use litellm_redis_testing::{RedisDeployment, Topology};
use rstest::rstest;
use support::connect;

const PYTHONS_PAYLOAD: &[u8] =
    br#"{"object_type": "litellm_proxymodeltable", "model_ids": ["m-1"]}"#;

async fn next(heard: &mut BoxStream<'static, Heard>, within: Duration) -> Heard {
    tokio::time::timeout(within, heard.next())
        .await
        .expect("the subscription should report something in time")
        .expect("the subscription never ends")
}

/// The first thing heard that is not `Lost`, since resubscribing can fail a few times while Redis
/// starts.
async fn past_lost(heard: &mut BoxStream<'static, Heard>) -> Heard {
    futures_util::stream::unfold(heard, async |heard| {
        Some((next(heard, Duration::from_secs(90)).await, heard))
    })
    .filter(|event| std::future::ready(*event != Heard::Lost))
    .boxed()
    .next()
    .await
    .unwrap()
}

#[rstest]
#[case::without_a_namespace(None)]
#[case::with_a_namespace(Some("prod-eu"))]
#[tokio::test]
async fn a_change_published_on_the_channel_python_publishes_on_is_heard(
    #[case] namespace: Option<&str>,
    #[values(Topology::Standalone, Topology::Cluster)] topology: Topology,
) {
    let deployment = RedisDeployment::start(topology).await.unwrap();
    let client = connect(&deployment);
    let mut heard = config_sync::subscribe(&client, namespace);
    let subscribed = next(&mut heard, Duration::from_secs(10)).await;

    client
        .publish(
            &channel(namespace),
            &[b"not json".to_vec(), PYTHONS_PAYLOAD.to_vec()],
        )
        .await
        .unwrap();

    assert_eq!(subscribed, Heard::Subscribed);
    assert_eq!(
        next(&mut heard, Duration::from_secs(5)).await,
        Heard::Change(ConfigChange {
            object_type: "litellm_proxymodeltable".to_owned(),
            model_ids: vec!["m-1".to_owned()],
        })
    );
}

#[rstest]
#[tokio::test]
async fn a_restarted_redis_is_heard_as_lost_then_subscribed_again() {
    let deployment = RedisDeployment::start(Topology::Standalone).await.unwrap();
    let client = connect(&deployment);
    let mut heard = config_sync::subscribe(&client, None);
    let first = next(&mut heard, Duration::from_secs(10)).await;

    deployment.restart().await.unwrap();
    let lost = next(&mut heard, Duration::from_secs(30)).await;
    let again = past_lost(&mut heard).await;
    client
        .publish(&channel(None), &[PYTHONS_PAYLOAD.to_vec()])
        .await
        .unwrap();

    assert_eq!(
        [first, lost, again],
        [Heard::Subscribed, Heard::Lost, Heard::Subscribed]
    );
    assert!(matches!(
        next(&mut heard, Duration::from_secs(5)).await,
        Heard::Change(_)
    ));
}
