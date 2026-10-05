#![cfg(feature = "schema")]

use litellm_traces::response::TraceSQLResponse;
use litellm_traces::schema::response_schemas;
use rstest::rstest;
use serde_json::{Value, json};

#[rstest]
fn sql_response_serializes_only_data() {
    let response = TraceSQLResponse {
        data: vec![
            serde_json::from_value(json!({
                "answer": 42,
                "nested": {"items": [true, null, "9007199254740993"]}
            }))
            .unwrap(),
        ],
    };

    assert_eq!(
        serde_json::to_value(response).unwrap(),
        json!({"data": [{"answer": 42, "nested": {"items": [true, null, "9007199254740993"]}}]})
    );
}

#[rstest]
fn sql_response_schema_requires_data_and_leaves_rows_open() {
    let schemas = response_schemas();
    let schema: Value = serde_json::to_value(&schemas["TraceSQLResponse"]).unwrap();

    assert_eq!(schema["additionalProperties"], false);
    assert_eq!(schema["required"], json!(["data"]));
    assert_eq!(schema["properties"].as_object().unwrap().len(), 1);
    assert!(
        schema["properties"]
            .as_object()
            .unwrap()
            .contains_key("data")
    );
    assert_eq!(schema["properties"]["data"]["type"], "array");
    assert_eq!(schema["properties"]["data"]["items"]["type"], "object");
    assert_ne!(
        schema["properties"]["data"]["items"]["additionalProperties"],
        false
    );
}
