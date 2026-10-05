#![cfg(feature = "schema")]

use litellm_traces::{
    api::{
        TraceHistogramRequest, TraceListRequest, TraceNoQueryRequest, TraceQueryRequest,
        TraceSQLResponse, TraceSpanPageRequest, TraceValuesRequest,
    },
    schema,
};
use rstest::rstest;
use serde_json::{Value, json};

#[rstest]
#[case::default(json!({}), true)]
#[case::one(json!({"page_size":1}), true)]
#[case::maximum(json!({"page_size":500}), true)]
#[case::zero(json!({"page_size":0}), false)]
#[case::over_maximum(json!({"page_size":501}), false)]
#[case::wrong_sort(json!({"sort_by":"spend"}), false)]
#[case::unknown_field(json!({"limit":10}), false)]
#[case::long_query(json!({"q":"x".repeat(1001)}), false)]
#[case::long_cursor(json!({"cursor":"x".repeat(513)}), false)]
fn list_contract_validates_the_same_request_in_rust_and_json_schema(
    #[case] input: Value,
    #[case] valid: bool,
) {
    let schemas = schema::api_schemas();
    assert_eq!(
        serde_json::from_value::<TraceListRequest>(input.clone()).is_ok(),
        valid
    );
    assert_eq!(
        jsonschema::is_valid(schemas["TraceListRequest"].as_value(), &input),
        valid
    );
}

#[rstest]
#[case::histogram_zero("TraceHistogramRequest",json!({"buckets":0}),false)]
#[case::histogram_max("TraceHistogramRequest",json!({"buckets":240}),true)]
#[case::histogram_over_max("TraceHistogramRequest",json!({"buckets":241}),false)]
#[case::values_zero("TraceValuesRequest",json!({"limit":0}),false)]
#[case::values_max("TraceValuesRequest",json!({"limit":100}),true)]
#[case::values_over_max("TraceValuesRequest",json!({"limit":101}),false)]
fn aggregation_contract_validates_limits(
    #[case] name: &str,
    #[case] input: Value,
    #[case] valid: bool,
) {
    let decoded = match name {
        "TraceHistogramRequest" => {
            serde_json::from_value::<TraceHistogramRequest>(input.clone()).is_ok()
        }
        "TraceValuesRequest" => serde_json::from_value::<TraceValuesRequest>(input.clone()).is_ok(),
        _ => unreachable!(),
    };
    let schemas = schema::api_schemas();
    assert_eq!(decoded, valid);
    assert_eq!(
        jsonschema::is_valid(schemas[name].as_value(), &input),
        valid
    );
}

#[rstest]
fn sql_contract_preserves_native_bind_values_and_engine_response_fields() {
    let request = json!({"sql":"SELECT {value:String}","params":{"value":"a'b", "flag":true, "ids":["one","two"],"missing":null,"large":u64::MAX}});
    let parsed: TraceQueryRequest = serde_json::from_value(request.clone()).unwrap();
    assert_eq!(serde_json::to_value(parsed).unwrap(), request);
    let response = json!({"meta":[{"name":"x","type":"UInt64","source":"traces"}],"data":[{"x":u64::MAX.to_string(),"nested":{"flags":[true,null]}}],"rows":1,"statistics":{"elapsed":0.01,"rows_read":"1","bytes_read":20,"extra_stat":42},"extra_field":[1,2]});
    let parsed: TraceSQLResponse = serde_json::from_value(response.clone()).unwrap();
    assert_eq!(serde_json::to_value(parsed).unwrap(), response);
    assert!(jsonschema::is_valid(
        schema::api_schemas()["TraceSQLResponse"].as_value(),
        &response
    ));
}

#[rstest]
#[case::spans_default("TraceSpanPageRequest",json!({}),true)]
#[case::spans_zero("TraceSpanPageRequest",json!({"page_size":0}),false)]
#[case::spans_maximum("TraceSpanPageRequest",json!({"page_size":500}),true)]
#[case::spans_over_maximum("TraceSpanPageRequest",json!({"page_size":501}),false)]
#[case::no_query("TraceNoQueryRequest",json!({}),true)]
#[case::old_trace_ref("TraceNoQueryRequest",json!({"trace_ref":"other"}),false)]
#[case::sql_empty("TraceQueryRequest",json!({"sql":""}),false)]
#[case::sql_unknown_field("TraceQueryRequest",json!({"sql":"SELECT 1","cursor":"x"}),false)]
fn collection_and_sql_contracts_validate_requests(
    #[case] name: &str,
    #[case] input: Value,
    #[case] valid: bool,
) {
    let decoded = match name {
        "TraceSpanPageRequest" => {
            serde_json::from_value::<TraceSpanPageRequest>(input.clone()).is_ok()
        }
        "TraceNoQueryRequest" => {
            serde_json::from_value::<TraceNoQueryRequest>(input.clone()).is_ok()
        }
        "TraceQueryRequest" => serde_json::from_value::<TraceQueryRequest>(input.clone()).is_ok(),
        _ => unreachable!(),
    };
    let schemas = schema::api_schemas();
    assert_eq!(decoded, valid);
    assert_eq!(
        jsonschema::is_valid(schemas[name].as_value(), &input),
        valid
    );
}
