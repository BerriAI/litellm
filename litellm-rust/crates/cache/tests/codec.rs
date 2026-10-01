use std::collections::BTreeMap;

use litellm_cache::{CacheCodec, Error, JsonCodec};
use rstest::rstest;
use serde::{Deserialize, Serialize};
use serde_json::json;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
struct RoutingState {
    deployment: String,
    cooldown_seconds: u64,
}

#[rstest]
fn json_codec_round_trips_typed_domain_values() {
    let codec = JsonCodec::<RoutingState>::new();
    let value = RoutingState {
        deployment: "deployment-a".into(),
        cooldown_seconds: 30,
    };
    let bytes = codec.encode(&value).unwrap();
    assert_eq!(codec.decode(&bytes).unwrap(), value);
    assert_eq!(
        serde_json::from_slice::<serde_json::Value>(&bytes).unwrap(),
        json!({"deployment": "deployment-a", "cooldown_seconds": 30})
    );
}

#[rstest]
#[case::malformed(b"not json")]
#[case::wrongly_typed(br#"{"deployment":12}"#)]
fn json_codec_rejects_malformed_and_wrongly_typed_entries(#[case] bytes: &[u8]) {
    let codec = JsonCodec::<RoutingState>::new();
    assert_eq!(codec.decode(bytes).unwrap_err(), Error::InvalidEntry);
}

#[rstest]
fn json_codec_propagates_encoding_errors() {
    let codec = JsonCodec::<BTreeMap<(u8, u8), String>>::new();
    let value = BTreeMap::from([((1, 2), "invalid JSON object key".into())]);
    assert_eq!(codec.encode(&value).unwrap_err(), Error::InvalidEntry);
}
