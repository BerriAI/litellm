use litellm_llms_types::formats::ocr::{LiteLLMOcrResponse, OcrDocument, OcrPage};
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
fn geometry_tables_and_key_value_pairs_expose_fields_without_changing_native_data() {
    let wire = json!({"model":"model","pages":[{"index":0,"markdown":"","images":[{"image_base64":null,"bbox":{"x":1,"y":0.5,"width":null,"future":[1,null]}}],"dimensions":null}],
        "document_annotation":{"schema_defined":[1,null]},"usage_info":null,"content":null,
        "tables":[{"rowCount":2,"columnCount":1,"cells":[{"rowIndex":0,"columnIndex":0,"content":"cell","boundingRegions":[{"pageNumber":1,"polygon":[0,0.5,1,1]}],"spans":[{"offset":0,"length":4}]}],"future":null}],
        "keyValuePairs":[{"key":{"content":"name"},"value":{"content":"value"},"confidence":1}],"object":"ocr"
    });
    let parsed: LiteLLMOcrResponse = serde_json::from_value(wire.clone()).unwrap();
    let bbox = parsed.pages[0].images.as_ref().unwrap()[0]
        .bbox
        .as_ref()
        .unwrap();
    assert_eq!(
        bbox.x,
        Some(litellm_llms_types::recognized::Recognized::Known(
            serde_json::Number::from(1)
        ))
    );
    let table = &parsed.tables.as_ref().unwrap()[0];
    assert_eq!(
        table.row_count,
        Some(litellm_llms_types::recognized::Recognized::Known(2))
    );
    let cell = table.cells.as_ref().unwrap().known().unwrap()[0]
        .known()
        .unwrap();
    assert_eq!(
        cell.content,
        Some(litellm_llms_types::recognized::Recognized::Known(
            "cell".into()
        ))
    );
    assert_eq!(
        cell.spans.as_ref().unwrap().known().unwrap()[0]
            .known()
            .unwrap()
            .length,
        Some(litellm_llms_types::recognized::Recognized::Known(4))
    );
    assert_eq!(
        parsed.key_value_pairs.as_ref().unwrap()[0]
            .value
            .as_ref()
            .unwrap()
            .known()
            .unwrap()
            .content,
        Some(litellm_llms_types::recognized::Recognized::Known(
            "value".into()
        ))
    );
    assert_eq!(parsed.into_json(), wire);
}

#[rstest]
fn partially_typed_ocr_objects_keep_malformed_and_unknown_fields() {
    let fields = json!({"bbox":{"x":"unknown","width":null,"future":true},"tables":[{"cells":false,"rowCount":null}],"keyValuePairs":[{"key":17,"confidence":"unknown"}]});
    let response: LiteLLMOcrResponse = serde_json::from_value(json!({"model":"model","pages":[],"tables":fields["tables"],"keyValuePairs":fields["keyValuePairs"]})).unwrap();
    let serialized = response.into_json();
    assert_eq!(serialized["tables"], fields["tables"]);
    assert_eq!(serialized["keyValuePairs"], fields["keyValuePairs"]);
    let image: litellm_llms_types::formats::ocr::OcrPageImage =
        serde_json::from_value(json!({"bbox":fields["bbox"]})).unwrap();
    assert_eq!(serde_json::to_value(image).unwrap()["bbox"], fields["bbox"]);
}
