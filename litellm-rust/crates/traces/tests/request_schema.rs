#![cfg(feature = "schema")]

use litellm_traces::request::{
    TraceDetailRequest, TraceErrorPageRequest, TraceListRequest, TraceQueryRequest,
    TraceSpanRequest,
};
use litellm_traces::schema::request_schemas;
use rstest::rstest;
use serde_json::json;

#[rstest]
#[case::list("TraceListRequest")]
#[case::detail("TraceDetailRequest")]
#[case::span("TraceSpanRequest")]
#[case::error_page("TraceErrorPageRequest")]
fn get_request_schemas_ignore_unknown_fields(#[case] name: &str) {
    let schemas = request_schemas();
    let schema = serde_json::to_value(&schemas[name]).unwrap();
    assert_ne!(schema["additionalProperties"], false);
}

#[rstest]
fn query_request_schema_rejects_unknown_fields() {
    let schemas = request_schemas();
    let schema = serde_json::to_value(&schemas["TraceQueryRequest"]).unwrap();
    assert_eq!(schema["additionalProperties"], false);
}

#[rstest]
fn request_schemas_preserve_explicit_constraints() {
    let schemas = request_schemas();
    let detail = serde_json::to_value(&schemas["TraceDetailRequest"]).unwrap();
    let list = serde_json::to_value(&schemas["TraceListRequest"]).unwrap();
    let span = serde_json::to_value(&schemas["TraceSpanRequest"]).unwrap();
    let error_page = serde_json::to_value(&schemas["TraceErrorPageRequest"]).unwrap();
    let query = serde_json::to_value(&schemas["TraceQueryRequest"]).unwrap();

    assert_eq!(detail["properties"]["page_size"]["minimum"], 1);
    assert_eq!(detail["properties"]["page_size"]["maximum"], 500);
    assert_eq!(list["properties"]["cursor"]["maxLength"], 512);
    assert_eq!(detail["properties"]["cursor"]["maxLength"], 512);
    assert_eq!(error_page["properties"]["cursor"]["maxLength"], 512);
    assert!(list["properties"]["start_ms"].get("minimum").is_none());
    assert!(list["properties"]["start_ms"].get("maximum").is_none());
    assert!(list["properties"]["end_ms"].get("minimum").is_none());
    assert!(list["properties"]["end_ms"].get("maximum").is_none());
    assert_eq!(detail["properties"]["trace_ref"]["default"], "");
    assert_eq!(span["properties"]["trace_ref"]["default"], "");
    assert_eq!(error_page["properties"]["trace_ref"]["default"], "");
    assert_eq!(query["required"], json!(["sql"]));
    assert_eq!(
        list["properties"]["start_ms"]["description"],
        "Window start, unix ms. Default: 24h ago"
    );
    assert_eq!(
        list["properties"]["end_ms"]["description"],
        "Window end, unix ms. Default: now"
    );
}

#[rstest]
fn request_models_deserialize_defaults_and_null_cursors() {
    let list: TraceListRequest = serde_json::from_value(json!({})).unwrap();
    let detail: TraceDetailRequest = serde_json::from_value(json!({})).unwrap();
    let span: TraceSpanRequest = serde_json::from_value(json!({})).unwrap();
    let error_page: TraceErrorPageRequest = serde_json::from_value(json!({})).unwrap();

    assert!(list.start_ms.is_none());
    assert!(list.end_ms.is_none());
    assert!(list.cursor.is_none());
    assert_eq!(detail.trace_ref, "");
    assert!(detail.cursor.is_none());
    assert!(detail.page_size.is_none());
    assert_eq!(span.trace_ref, "");
    assert_eq!(error_page.trace_ref, "");
    assert!(error_page.cursor.is_none());

    let null_cursor: TraceListRequest = serde_json::from_value(json!({"cursor": null})).unwrap();
    assert!(null_cursor.cursor.is_none());
}

#[rstest]
fn get_request_models_ignore_unknown_fields_and_query_model_rejects_them() {
    assert!(serde_json::from_value::<TraceListRequest>(json!({"unknown": true})).is_ok());
    assert!(serde_json::from_value::<TraceDetailRequest>(json!({"unknown": true})).is_ok());
    assert!(serde_json::from_value::<TraceSpanRequest>(json!({"unknown": true})).is_ok());
    assert!(serde_json::from_value::<TraceErrorPageRequest>(json!({"unknown": true})).is_ok());
    assert!(
        serde_json::from_value::<TraceQueryRequest>(json!({"sql": "SELECT 1", "unknown": true}))
            .is_err()
    );
    assert!(serde_json::from_value::<TraceQueryRequest>(json!({})).is_err());
}
