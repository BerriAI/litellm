use std::collections::BTreeMap;

use litellm_cache::{CacheCodec, Error, JsonCodec};
use serde::{Deserialize, Serialize};
use serde_json::json;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
struct RoutingState {
    deployment: String,
    cooldown_seconds: u64,
}

#[test]
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

#[test]
fn json_codec_rejects_malformed_and_wrongly_typed_entries() {
    let codec = JsonCodec::<RoutingState>::new();
    for bytes in [b"not json".as_slice(), br#"{"deployment":12}"#.as_slice()] {
        assert_eq!(codec.decode(bytes).unwrap_err(), Error::InvalidEntry);
    }
}

#[test]
fn json_codec_propagates_encoding_errors() {
    let codec = JsonCodec::<BTreeMap<(u8, u8), String>>::new();
    let value = BTreeMap::from([((1, 2), "invalid JSON object key".into())]);
    assert_eq!(codec.encode(&value).unwrap_err(), Error::InvalidEntry);
}
