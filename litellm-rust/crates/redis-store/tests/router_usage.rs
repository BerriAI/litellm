mod support;

use std::time::Duration;

use litellm_redis_client::{Expiry, Ttl};
use litellm_redis_store::{
    Number,
    router_usage::{COUNTER_TTL, Counter, Kind, POLICY, RouterUsage},
};
use litellm_redis_testing::{RedisDeployment, Topology};
use rstest::rstest;
use support::{connect, usage, within};

fn tpm_and_rpm(model: Option<&str>) -> [Counter; 2] {
    [Kind::Tpm, Kind::Rpm].map(|kind| Counter {
        deployment_id: "dep".to_owned(),
        model: model.map(str::to_owned),
        minute: "12-34".to_owned(),
        kind,
    })
}

#[rstest]
#[tokio::test]
async fn reads_counters_python_incremented_under_mains_keys(
    #[values(Topology::Standalone, Topology::Cluster)] topology: Topology,
) {
    let deployment = RedisDeployment::start(topology).await.unwrap();
    let raw = connect(&deployment);
    let store = RouterUsage::new(Some(connect(&deployment)), POLICY);
    let cases = usage();
    for (case, delta) in cases.iter().zip([7.75, 3.0]) {
        raw.incr_by_float(case.key, delta, Expiry::Refresh(COUNTER_TTL))
            .await
            .unwrap();
    }

    assert_eq!(
        store
            .get_many(&cases.map(|case| case.counter))
            .await
            .unwrap(),
        vec![Some(Number::Float(7.75)), Some(Number::Int(3))]
    );
}

#[rstest]
#[tokio::test]
async fn deployment_increments_write_incrbyfloat_totals_with_a_refreshed_minute_window(
    #[values(Topology::Standalone, Topology::Cluster)] topology: Topology,
) {
    let deployment = RedisDeployment::start(topology).await.unwrap();
    let raw = connect(&deployment);
    let writer = RouterUsage::new(Some(connect(&deployment)), POLICY);
    let reader = RouterUsage::new(Some(connect(&deployment)), POLICY);
    let [case, _] = usage();
    let model = case.counter.model.as_deref();
    let [tpm_key, rpm_key] =
        ["tpm", "rpm"].map(|kind| case.key.replace(":tpm:", &format!(":{kind}:")));

    writer
        .increment_deployment("dep", model, "12-34", Number::Float(5.5), 1)
        .await
        .unwrap();
    raw.expire(&tpm_key, Duration::from_secs(5)).await.unwrap();
    writer
        .increment_deployment("dep", model, "12-34", Number::Int(2), 1)
        .await
        .unwrap();

    let written: [Option<String>; 2] = [
        raw.get(&tpm_key).await.unwrap(),
        raw.get(&rpm_key).await.unwrap(),
    ];
    let Ttl::Expires(remaining) = raw.ttl(&tpm_key).await.unwrap() else {
        panic!("{tpm_key} has no TTL");
    };
    assert_eq!(written, [Some("7.5".to_owned()), Some("2".to_owned())]);
    assert!(within(remaining, 60), "{remaining:?}");
    assert_eq!(
        reader.get_many(&tpm_and_rpm(model)).await.unwrap(),
        vec![Some(Number::Float(7.5)), Some(Number::Int(2))]
    );
    assert_eq!(
        writer.get_many(&tpm_and_rpm(model)).await.unwrap(),
        vec![Some(Number::Int(7)), Some(Number::Int(2))]
    );
}

#[rstest]
#[tokio::test]
async fn a_request_increment_keeps_the_window_python_started(
    #[values(Topology::Standalone, Topology::Cluster)] topology: Topology,
) {
    let deployment = RedisDeployment::start(topology).await.unwrap();
    let raw = connect(&deployment);
    let store = RouterUsage::new(Some(connect(&deployment)), POLICY);
    let [_, case] = usage();
    let key = case.key;
    raw.incr_by_float(key, 1.0, Expiry::KeepExisting(Duration::from_secs(20)))
        .await
        .unwrap();

    let total = store
        .increment(&case.counter, Number::Int(1), COUNTER_TTL)
        .await
        .unwrap();
    let Ttl::Expires(remaining) = raw.ttl(key).await.unwrap() else {
        panic!("{key} has no TTL");
    };

    assert_eq!(total, Number::Float(2.0));
    assert!(within(remaining, 20), "{remaining:?}");
}

#[rstest]
#[tokio::test]
async fn without_redis_counters_accumulate_in_memory_until_flushed() {
    let store = RouterUsage::new(None, POLICY);
    let counters = tpm_and_rpm(Some("model"));

    store
        .increment_deployment("dep", Some("model"), "12-34", Number::Int(2), 1)
        .await
        .unwrap();
    store
        .increment_deployment("dep", Some("model"), "12-34", Number::Float(5.5), 1)
        .await
        .unwrap();
    let counted = store.get_many(&counters).await.unwrap();
    store.flush_local_counters().unwrap();

    assert_eq!(
        counted,
        vec![Some(Number::Float(7.5)), Some(Number::Int(2))]
    );
    assert_eq!(store.get_many(&counters).await.unwrap(), vec![None, None]);
}

#[rstest]
#[tokio::test]
async fn a_failed_pipeline_keeps_the_memory_counts_as_integers() {
    let deployment = RedisDeployment::start(Topology::Standalone).await.unwrap();
    let store = RouterUsage::new(Some(connect(&deployment)), POLICY);
    deployment.stop().await.unwrap();

    store
        .increment_deployment("dep", Some("model"), "12-34", Number::Float(5.75), 1)
        .await
        .unwrap();

    assert_eq!(
        store.get_many(&tpm_and_rpm(Some("model"))).await.unwrap(),
        vec![Some(Number::Int(5)), Some(Number::Int(1))]
    );
}
