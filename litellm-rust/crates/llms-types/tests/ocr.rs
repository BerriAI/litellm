use litellm_llms_types::formats::ocr::{LiteLLMOcrResponse, OcrBoundingBox, OcrDocument, OcrPage};
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
fn bounding_box_exposes_corner_coordinates_and_keeps_extensions() {
    let wire = json!({"top_left_x":1,"top_left_y":2.5,"bottom_right_x":30,"bottom_right_y":40,"future":true});
    let bounds: OcrBoundingBox = serde_json::from_value(wire.clone()).unwrap();
    assert_eq!(bounds.top_left_x, Some(1.into()));
    assert_eq!(
        bounds
            .top_left_y
            .as_ref()
            .and_then(serde_json::Number::as_f64),
        Some(2.5)
    );
    assert_eq!(bounds.bottom_right_x, Some(30.into()));
    assert_eq!(bounds.bottom_right_y, Some(40.into()));
    assert_eq!(Value::Object(bounds.extra.clone()), json!({"future":true}));
    assert_eq!(serde_json::to_value(bounds).unwrap(), wire);
}

#[rstest]
fn partial_bounding_box_omits_null_corners() {
    let bounds: OcrBoundingBox =
        serde_json::from_value(json!({"top_left_x":null,"bottom_right_y":4})).unwrap();
    assert!(bounds.top_left_x.is_none());
    assert_eq!(
        serde_json::to_value(bounds).unwrap(),
        json!({"bottom_right_y":4})
    );
}

#[rstest]
#[case::string_corner(json!({"top_left_x":"1"}))]
#[case::array_corner(json!({"bottom_right_y":[4]}))]
fn bounding_box_rejects_non_numeric_corners(#[case] wire: Value) {
    assert!(serde_json::from_value::<OcrBoundingBox>(wire).is_err());
}
