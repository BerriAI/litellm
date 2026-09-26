use crate::error::CostError;
use serde_json::Value;

use crate::non_token::{OcrRates, OcrUsage, calculate_ocr_with_tables};
use crate::pricing::Rate;

fn rate(model_info: &Value, key: &str) -> Rate {
    model_info
        .get(key)
        .and_then(Value::as_f64)
        .map_or(Rate::Missing, Rate::Value)
}

fn rates(model_info: &Value) -> OcrRates {
    OcrRates {
        per_credit: rate(model_info, "ocr_cost_per_credit"),
        per_page: rate(model_info, "ocr_cost_per_page"),
        per_annotation_page: rate(model_info, "annotation_cost_per_page"),
    }
}

fn value(rate: Rate) -> Option<f64> {
    match rate {
        Rate::Value(value) => Some(value),
        Rate::Missing | Rate::Null | Rate::Invalid => None,
    }
}

pub fn ocr_cost(
    response: &Value,
    deployment_info: Option<&Value>,
    published_info: Option<&Value>,
) -> Result<(f64, f64), CostError> {
    let usage = response
        .get("usage_info")
        .and_then(Value::as_object)
        .ok_or(CostError::MissingUsage)?;
    let pages = usage.get("pages_processed").and_then(Value::as_u64);
    let annotation_pages = usage
        .get("pages_processed_annotation")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let credits = usage.get("credits").and_then(Value::as_f64);
    let tables: Vec<OcrRates> = [deployment_info, published_info]
        .into_iter()
        .flatten()
        .map(rates)
        .collect();
    let first = |field: fn(OcrRates) -> Rate| tables.iter().find_map(|table| value(field(*table)));
    let credit_rate = first(|table| table.per_credit);
    let page_rate = first(|table| table.per_page);
    let annotation_rate = first(|table| table.per_annotation_page).or(page_rate);
    let bills_credits = credits.is_some() && credit_rate.is_some();
    let bills_annotations = annotation_pages > 0 && annotation_rate.is_some();
    if !bills_credits && pages.is_none() && !bills_annotations {
        if credit_rate.is_some() || page_rate.is_none() {
            return Ok((0.0, 0.0));
        }
        return Err(CostError::MissingPages);
    }
    let cost = calculate_ocr_with_tables(
        &tables,
        OcrUsage {
            credits,
            pages: pages.unwrap_or(0),
            annotation_pages,
        },
    )?;
    Ok((cost.total, 0.0))
}
