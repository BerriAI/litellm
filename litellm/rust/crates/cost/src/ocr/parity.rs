//! Parity with main's `ocr_cost` and `ocr_batch_cost`. `fixtures/ocr_parity.json` was captured once from main's
//! Python: each case holds the cost fields `get_model_info` returned, the deployment's own pricing, the reported
//! `usage_info` and main's `(prompt, completion)`, or that it raised. The adapter raises on `MissingUsage` and
//! answers any other unpriced call with `(0.0, 0.0)`, as main did

use rstest::rstest;
use serde_json::Value;

use super::{OcrQuery, price};
use crate::query::{Header, PriceSource};
use crate::receipt::{PriceOutcome, Unpriced};
use crate::testing::{TestSource, expected_sides, header, row_ref, units_row};

const FIXTURE: &str = include_str!("../../fixtures/ocr_parity.json");
const DEPLOYMENT: &str = "deployment";

fn case(name: &str) -> Value {
    let fixture: Value = serde_json::from_str(FIXTURE).unwrap();
    fixture["cases"]
        .as_array()
        .unwrap()
        .iter()
        .find(|case| case["name"] == name)
        .unwrap_or_else(|| panic!("no fixture case {name}"))
        .clone()
}

#[rstest]
#[case::pages("pages")]
#[case::pages_and_annotations("pages_and_annotations")]
#[case::annotations_without_pages("annotations_without_pages")]
#[case::no_pages_on_page_priced_row("no_pages_on_page_priced_row")]
#[case::deployment_page_rate("deployment_page_rate")]
#[case::deployment_annotation_rate("deployment_annotation_rate")]
#[case::annotations_at_the_page_rate("annotations_at_the_page_rate")]
#[case::credits("credits")]
#[case::credit_priced_without_pages("credit_priced_without_pages")]
#[case::credit_priced_without_credits("credit_priced_without_credits")]
#[case::unmapped("unmapped")]
#[case::unmapped_deployment_credits("unmapped_deployment_credits")]
#[case::batch_rates("batch_rates")]
#[case::batch_online_rates("batch_online_rates")]
#[case::batch_deployment_online_page_rate("batch_deployment_online_page_rate")]
#[case::batch_deployment_annotation_family("batch_deployment_annotation_family")]
#[case::batch_unmapped_deployment_page_rate("batch_unmapped_deployment_page_rate")]
#[case::batch_unmapped("batch_unmapped")]
fn an_ocr_result_costs_what_main_charged(#[case] name: &str) {
    let case = case(name);
    let model = case["model"].as_str().unwrap();
    let catalog = match case["row"].is_null() {
        true => TestSource::default(),
        false => TestSource::with_row(model, units_row(&case["row"])),
    };
    let (source, given) = match case["deployment"].is_null() {
        true => (catalog, Vec::new()),
        false => (
            catalog.and_row(DEPLOYMENT, units_row(&case["deployment"])),
            vec![row_ref(DEPLOYMENT)],
        ),
    };
    let usage = &case["usage"];
    let query = OcrQuery {
        header: Header {
            source: match given.is_empty() {
                true => PriceSource::Catalog,
                false => PriceSource::Rows { rows: given },
            },
            ..header(case["provider"].as_str().unwrap(), model)
        },
        pages: usage["pages_processed"].as_u64(),
        annotation_pages: usage["pages_processed_annotation"].as_u64().unwrap_or(0),
        credits: usage["credits"].as_f64(),
        batch: case["batch"].as_bool().unwrap(),
        catalog_deferred: false,
    };
    let sides = match price(&source, &query) {
        PriceOutcome::Priced(receipt) => Some((receipt.prompt.get(), receipt.completion.get())),
        PriceOutcome::Unpriced(Unpriced::MissingUsage { .. }) => None,
        PriceOutcome::Unpriced(_) => Some((0.0, 0.0)),
    };
    assert_eq!(sides, expected_sides(&case));
}
