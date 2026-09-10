use std::collections::BTreeSet;

use serde_json::{Map, Value};

use super::types::*;
use crate::ocr::error::{OcrRequestError, PagesError};

pub(super) fn validate(params: &Map<String, Value>) -> Result<(), OcrRequestError> {
    let Some(Value::Array(pages)) = params.get("pages") else {
        return Ok(());
    };
    if pages.iter().any(Value::is_boolean) {
        return Err(PagesError::BooleanIndex.into());
    }
    if pages
        .iter()
        .any(|page| page.is_number() && page.as_i64().is_none())
    {
        return Err(PagesError::IndexOutOfRange.into());
    }
    if !pages.iter().all(Value::is_i64) && !pages.iter().all(Value::is_string) {
        return Err(PagesError::MixedElementTypes.into());
    }
    Ok(())
}

fn pages_token_is_valid(token: &str) -> bool {
    let mut parts = token.split('-');
    let start = parts.next().unwrap_or_default();
    if start.is_empty() || !start.chars().all(|ch| ch.is_ascii_digit()) {
        return false;
    }
    match parts.next() {
        None => true,
        Some(end) => {
            !end.is_empty() && end.chars().all(|ch| ch.is_ascii_digit()) && parts.next().is_none()
        }
    }
}

fn normalize_pages_param(pages: PagesInput) -> Result<Option<NormalizedPages>, PagesError> {
    let normalized = match pages {
        PagesInput::ZeroBasedIndices(indices) => {
            if indices.is_empty() {
                return Ok(None);
            }
            indices
                .into_iter()
                .map(|page| {
                    if page < 0 {
                        return Err(PagesError::NegativeIndex);
                    }
                    page.checked_add(1).ok_or(PagesError::IndexOutOfRange)
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
    if !normalized.split(',').all(pages_token_is_valid) {
        return Err(PagesError::InvalidNativeRange);
    }
    Ok(Some(NormalizedPages(normalized)))
}

fn normalize_features_param(
    features: FeaturesInput,
) -> Result<Option<NormalizedFeatures>, OcrRequestError> {
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
    Ok(Some(NormalizedFeatures(normalized.join(","))))
}

pub(super) fn normalize(
    params: DocumentIntelligenceInputParams,
) -> Result<DocumentIntelligenceParams, OcrRequestError> {
    Ok(DocumentIntelligenceParams {
        pages: params
            .pages
            .map(normalize_pages_param)
            .transpose()?
            .flatten(),
        features: params
            .features
            .map(normalize_features_param)
            .transpose()?
            .flatten(),
        request_format: params.req_format.unwrap_or_default(),
    })
}
