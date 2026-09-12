use std::collections::BTreeSet;

use serde_json::{Map, Value};

use super::types::{
    DocumentIntelligenceInputParams, DocumentIntelligenceParams, FeaturesInput, PagesInput,
};
use crate::ocr::error::OcrRequestError;
use crate::ocr::prepare::ParsedProviderParams;

pub(crate) fn decode_input_params(
    params: Map<String, Value>,
    prefix: &str,
) -> Result<ParsedProviderParams<DocumentIntelligenceInputParams>, OcrRequestError> {
    if let Some(Value::Array(pages)) = params.get("pages") {
        if pages.iter().any(Value::is_boolean) {
            return Err(OcrRequestError::Pages("boolean page index".into()));
        }
        if pages
            .iter()
            .any(|page| page.is_number() && page.as_i64().is_none())
        {
            return Err(OcrRequestError::Pages("page index is out of range".into()));
        }
        if !pages.iter().all(Value::is_i64) && !pages.iter().all(Value::is_string) {
            return Err(OcrRequestError::Pages("mixed page element types".into()));
        }
    }
    crate::ocr::wire::decode_request_value(Value::Object(params), prefix)
}

pub(crate) fn map_ocr_params(
    params: DocumentIntelligenceInputParams,
) -> Result<DocumentIntelligenceParams, OcrRequestError> {
    Ok(DocumentIntelligenceParams {
        pages: params.pages.map(normalize_pages).transpose()?.flatten(),
        features: params
            .features
            .map(normalize_features)
            .transpose()?
            .flatten(),
    })
}

fn normalize_pages(pages: PagesInput) -> Result<Option<String>, OcrRequestError> {
    let normalized = match pages {
        PagesInput::ZeroBasedIndices(indices) => {
            if indices.is_empty() {
                return Ok(None);
            }
            indices
                .into_iter()
                .map(|page| {
                    if page < 0 {
                        return Err(OcrRequestError::Pages("negative page index".into()));
                    }
                    page.checked_add(1)
                        .ok_or_else(|| OcrRequestError::Pages("page index is out of range".into()))
                })
                .collect::<Result<BTreeSet<_>, _>>()?
                .into_iter()
                .map(|page| page.to_string())
                .collect::<Vec<_>>()
                .join(",")
        }
        PagesInput::NativeTokens(tokens) => {
            if tokens.is_empty() {
                return Ok(None);
            }
            tokens
                .iter()
                .map(|token| token.trim())
                .collect::<Vec<_>>()
                .join(",")
        }
        PagesInput::NativeRange(range) => range
            .split(',')
            .map(str::trim)
            .collect::<Vec<_>>()
            .join(","),
    };
    if !normalized.split(',').all(valid_page_token) {
        return Err(OcrRequestError::Pages("invalid native page range".into()));
    }
    Ok(Some(normalized))
}

fn valid_page_token(token: &str) -> bool {
    let mut parts = token.split('-');
    let start = parts.next().unwrap_or_default();
    if start.is_empty() || !start.chars().all(|character| character.is_ascii_digit()) {
        return false;
    }
    match parts.next() {
        None => true,
        Some(end) => {
            !end.is_empty()
                && end.chars().all(|character| character.is_ascii_digit())
                && parts.next().is_none()
        }
    }
}

fn normalize_features(features: FeaturesInput) -> Result<Option<String>, OcrRequestError> {
    let tokens = match features {
        FeaturesInput::Names(names) => names,
        FeaturesInput::CommaSeparated(names) => names.split(',').map(str::to_string).collect(),
    };
    if tokens.is_empty() {
        return Ok(None);
    }
    let normalized = tokens.iter().map(|token| token.trim()).collect::<Vec<_>>();
    if !normalized.iter().all(|token| {
        let Some((first, rest)) = token.as_bytes().split_first() else {
            return false;
        };
        first.is_ascii_alphabetic() && rest.iter().all(u8::is_ascii_alphanumeric)
    }) {
        return Err(OcrRequestError::Features);
    }
    Ok(Some(normalized.join(",")))
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use serde_json::{Value, json};

    use super::*;

    fn map(value: Value) -> Result<DocumentIntelligenceParams, OcrRequestError> {
        let fields = value.as_object().unwrap().clone();
        map_ocr_params(decode_input_params(fields, "optional_params")?.known)
    }

    #[test]
    fn input_params_retain_unknown_fields() {
        let parsed = decode_input_params(
            json!({
                "pages": [0],
                "future_ocr_option": true,
                "extra_body": {"provider_option": "value"}
            })
            .as_object()
            .unwrap()
            .clone(),
            "optional_params",
        )
        .unwrap();

        assert_eq!(
            parsed.known.pages,
            Some(PagesInput::ZeroBasedIndices(vec![0]))
        );
        assert_eq!(parsed.extra_params["future_ocr_option"], true);
        assert_eq!(
            parsed.extra_params["extra_body"],
            json!({"provider_option": "value"})
        );
        assert_eq!(
            serde_json::to_value(map_ocr_params(parsed.known).unwrap()).unwrap(),
            json!({"pages": "1", "features": null})
        );
    }

    #[rstest]
    #[case(json!(["keyValuePairs"]), "keyValuePairs")]
    #[case(json!(["keyValuePairs", "languages"]), "keyValuePairs,languages")]
    #[case(json!("keyValuePairs"), "keyValuePairs")]
    #[case(json!("keyValuePairs,languages"), "keyValuePairs,languages")]
    #[case(json!("keyValuePairs, languages"), "keyValuePairs,languages")]
    fn feature_mapping_matches_python(#[case] input: Value, #[case] expected: &str) {
        assert_eq!(
            map(json!({"features": input})).unwrap().features.as_deref(),
            Some(expected)
        );
    }

    #[rstest]
    #[case(json!("keyValuePairs&pages=9"))]
    #[case(json!("key value pairs"))]
    #[case(json!(""))]
    #[case(json!([1, 2]))]
    #[case(json!([["keyValuePairs"]]))]
    #[case(json!({"feature":"keyValuePairs"}))]
    #[case(json!(5))]
    fn invalid_feature_mapping_matches_python(#[case] input: Value) {
        assert!(map(json!({"features": input})).is_err());
    }

    #[test]
    fn empty_feature_list_is_omitted() {
        assert_eq!(map(json!({"features": []})).unwrap().features, None);
    }
}
