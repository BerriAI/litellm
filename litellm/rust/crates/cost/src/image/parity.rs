//! Parity with main's `default_image_cost_calculator` and the per-image calculators. `fixtures/image_parity.json` was
//! captured once from main's Python: each case holds the row the calculator read, the images and pixels it billed and
//! main's total

use rstest::rstest;

use super::{ImageCalculator, ImageQuery, price};
use crate::testing::{TestSource, fixture_case, header, priced, units_row};

const FIXTURE: &str = include_str!("../../fixtures/image_parity.json");

#[rstest]
#[case::azure_hd_tall("azure_hd_tall", ImageCalculator::Default)]
#[case::azure_standard_bare_model("azure_standard_bare_model", ImageCalculator::Default)]
#[case::aiml_by_name("aiml_by_name", ImageCalculator::Default)]
#[case::recraft_by_name("recraft_by_name", ImageCalculator::Default)]
#[case::recraft("recraft", ImageCalculator::PerImage)]
#[case::aiml("aiml", ImageCalculator::PerImage)]
#[case::runwayml("runwayml", ImageCalculator::PerImage)]
#[case::dashscope_input_images("dashscope_input_images", ImageCalculator::PerImageAndInput)]
fn an_image_call_costs_what_main_charged(#[case] name: &str, #[case] calculator: ImageCalculator) {
    let case = fixture_case(FIXTURE, name);
    let model = case["model"].as_str().unwrap();
    let source = TestSource::with_row(model, units_row(&case["row"]));
    let query = ImageQuery {
        header: header(case["provider"].as_str().unwrap(), model),
        calculator,
        images: case["images"].as_u64().unwrap(),
        pixels: case["pixels"].as_u64().unwrap_or(0),
        input_images: case["input_images"].as_u64().unwrap_or(0),
    };
    assert_eq!(
        Some(priced(price(&source, &query)).total),
        case["expected"].as_f64()
    );
}
