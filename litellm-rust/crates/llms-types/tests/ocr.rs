use litellm_llms_types::formats::ocr::{
    LiteLLMOcrResponse, OcrBoundingBox, OcrDocument, OcrKeyValuePair, OcrPage, OcrTable,
};
use rstest::rstest;
use serde_json::{Map, Value, json};

#[rstest]
#[case::missing_page_fields(json!({"pages": [{}]}))]
#[case::invalid_markdown(json!({"pages": [{"index": 0, "markdown": false}]}))]
#[case::invalid_image_bounds(json!({"pages": [{"index": 0, "markdown": "", "images": [{"bbox": []}]}]}))]
#[case::fractional_page_count(json!({"usage_info": {"pages_processed": 1.5}}))]
#[case::invalid_table(json!({"tables": [false]}))]
#[case::invalid_key_value_pair(json!({"keyValuePairs": [[]]}))]
#[case::invalid_native_response(json!({"provider_native_response": []}))]
fn normalized_response_rejects_invalid_shared_fields(#[case] fields: Value) {
    let payload: Map<String, Value> = json!({"model": "model", "pages": []})
        .as_object()
        .unwrap()
        .iter()
        .chain(fields.as_object().unwrap())
        .map(|(key, value)| (key.clone(), value.clone()))
        .collect();
    assert!(serde_json::from_value::<LiteLLMOcrResponse>(Value::Object(payload)).is_err());
}

#[rstest]
fn document_rejects_non_string_provider_fields() {
    assert!(
        serde_json::from_value::<OcrDocument>(json!({
            "type": "image_url", "image_url": "https://example.com/image", "detail": 42
        }))
        .is_err()
    );
}

#[rstest]
#[case::large_integer(json!("9007199254740993.0"), 9_007_199_254_740_993)]
#[case::signed_decimal(json!("+2.000"), 2)]
#[case::separator(json!("1_000"), 1000)]
#[case::boolean(json!(true), 1)]
#[case::integral_float(json!(2.0), 2)]
fn numeric_coercion_preserves_integer_precision(#[case] value: Value, #[case] expected: i64) {
    let page: OcrPage = serde_json::from_value(json!({"index": value, "markdown": ""})).unwrap();
    assert_eq!(page.index, expected);
    assert_eq!(
        serde_json::to_value(page).unwrap()["index"],
        json!(expected)
    );
}

#[rstest]
#[case::exponent(json!("1e2"))]
#[case::missing_integer(json!(".0"))]
#[case::missing_fraction(json!("2."))]
#[case::leading_separator(json!("_2"))]
#[case::repeated_separator(json!("2__0"))]
#[case::fractional_float(json!(2.5))]
#[case::null(json!(null))]
fn page_index_rejects_invalid_integers(#[case] value: Value) {
    assert!(serde_json::from_value::<OcrPage>(json!({"index": value, "markdown": ""})).is_err());
}

#[rstest]
#[case::document_url("document_url", "document_name", "application/pdf")]
#[case::image_url("image_url", "detail", "image/png")]
fn document_variants_preserve_provider_fields_when_rewriting_sources(
    #[case] kind: &str,
    #[case] field: &str,
    #[case] mime_type: &str,
    #[values(json!("kept"), Value::Null)] extra: Value,
) {
    let original = "https://example.com/input";
    let replacement = format!("data:{mime_type};base64,AA==");
    let document: OcrDocument =
        serde_json::from_value(json!({"type": kind, kind: original, field: extra})).unwrap();
    assert_eq!(document.source(), original);
    assert!(document.is_remote());
    let rewritten = document.with_source(replacement.clone());
    assert!(!rewritten.is_remote());
    assert_eq!(
        serde_json::to_value(rewritten).unwrap(),
        json!({"type": kind, kind: replacement, field: extra})
    );
}

#[rstest]
#[case::absent_native(None)]
#[case::present_native(Some(Map::from_iter([("native".into(), json!({"nested": [null, 1]}))])))]
fn response_serialization_preserves_extensions_and_native_presence(
    #[case] native: Option<Map<String, Value>>,
) {
    let response = LiteLLMOcrResponse {
        extra_fields: Map::from_iter([("provider_field".into(), json!("kept"))]),
        provider_native_response: native.clone(),
        ..LiteLLMOcrResponse::new("model", vec![])
    };
    let serialized = response.into_json();
    assert_eq!(serialized["provider_field"], "kept");
    assert_eq!(
        serialized.get("provider_native_response").cloned(),
        native.clone().map(Value::Object)
    );
    let decoded: LiteLLMOcrResponse = serde_json::from_value(serialized.clone()).unwrap();
    assert_eq!(decoded.provider_native_response, native);
    assert_eq!(decoded.into_json(), serialized);
}

#[rstest]
fn bounding_box_round_trips() {
    let wire = json!({"x":1.5,"y":2,"width":3,"height":4,"future":true});
    let bounds: OcrBoundingBox = serde_json::from_value(wire.clone()).unwrap();
    assert_eq!(
        bounds.x.as_ref().and_then(serde_json::Number::as_f64),
        Some(1.5)
    );
    assert_eq!(bounds.width, Some(3.into()));
    assert_eq!(bounds.extra["future"], json!(true));
    assert_eq!(serde_json::to_value(bounds).unwrap(), wire);
}

#[rstest]
fn table_round_trips_nested_cells_and_regions() {
    let wire = json!({
        "rowCount":1,
        "columnCount":1,
        "cells":[{"rowIndex":0,"columnIndex":0,"content":"value","spans":[{"offset":0,"length":5}]}],
        "boundingRegions":[{"pageNumber":1,"polygon":[0,0,10,10]}],
        "spans":[{"offset":0,"length":5}],
        "content":"value"
    });
    let table: OcrTable = serde_json::from_value(wire.clone()).unwrap();
    assert_eq!(table.row_count, Some(1));
    assert_eq!(table.column_count, Some(1));
    let cell = &table.cells.as_ref().unwrap()[0];
    assert_eq!(cell.row_index, Some(0));
    assert_eq!(cell.column_index, Some(0));
    assert_eq!(cell.content.as_deref(), Some("value"));
    assert_eq!(cell.spans.as_ref().unwrap()[0].length, Some(5));
    let region = &table.bounding_regions.as_ref().unwrap()[0];
    assert_eq!(region.page_number, Some(1));
    assert_eq!(
        region.polygon.as_ref().unwrap(),
        &vec![0.into(), 0.into(), 10.into(), 10.into()]
    );
    assert_eq!(serde_json::to_value(table).unwrap(), wire);
}

#[rstest]
fn key_value_pair_round_trips_nested_elements() {
    let wire = json!({
        "key":{"content":"Name","boundingRegions":[{"pageNumber":1,"polygon":[0,0,1,1]}]},
        "value":{"content":"Ada","spans":[{"offset":5,"length":3}]},
        "confidence":0.99
    });
    let pair: OcrKeyValuePair = serde_json::from_value(wire.clone()).unwrap();
    let key = pair.key.as_ref().unwrap();
    let value = pair.value.as_ref().unwrap();
    assert_eq!(key.content.as_deref(), Some("Name"));
    assert_eq!(
        key.bounding_regions.as_ref().unwrap()[0].page_number,
        Some(1)
    );
    assert_eq!(value.content.as_deref(), Some("Ada"));
    assert_eq!(value.spans.as_ref().unwrap()[0].offset, Some(5));
    assert_eq!(serde_json::to_value(pair).unwrap(), wire);
}

#[rstest]
#[case::bad_cell(json!({"cells":[{"rowIndex":"first"}]}))]
#[case::negative_count(json!({"rowCount":-1}))]
#[case::bad_polygon(json!({"boundingRegions":[{"polygon":["zero"]}]}))]
#[case::bad_span(json!({"spans":[{"length":-1}]}))]
fn typed_tables_reject_malformed_nested_fields(#[case] wire: Value) {
    assert!(serde_json::from_value::<OcrTable>(wire).is_err());
}

#[rstest]
fn partial_table_omits_null_optionals_and_keeps_extensions() {
    let table: OcrTable =
        serde_json::from_value(json!({"cells":null,"rowCount":null,"future":null})).unwrap();
    assert!(table.cells.is_none());
    assert!(table.row_count.is_none());
    assert_eq!(serde_json::to_value(table).unwrap(), json!({"future":null}));
}
