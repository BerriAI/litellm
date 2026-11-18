//! Parity with main's `video_generation_cost`. `fixtures/video_parity.json` was captured once from main's Python: each
//! case holds the cost fields `get_model_info` returned, the reported length and resolution, and main's total. The
//! adapter answers a video no rate prices with 0.0, as main did

use rstest::rstest;

use super::{VideoQuery, price};
use crate::receipt::PriceOutcome;
use crate::testing::{TestSource, fixture_case, header, units_row};

const FIXTURE: &str = include_str!("../../fixtures/video_parity.json");

#[rstest]
#[case::sora_per_second("sora_per_second")]
#[case::sora_pro_resolution_rate("sora_pro_resolution_rate")]
#[case::clip_cell("clip_cell")]
#[case::clip_cell_rounds_half_to_even("clip_cell_rounds_half_to_even")]
#[case::clip_without_cell_is_unpriced("clip_without_cell_is_unpriced")]
#[case::resolution_rate("resolution_rate")]
#[case::resolution_without_rate("resolution_without_rate")]
#[case::per_video("per_video")]
#[case::measured_length("measured_length")]
fn a_video_costs_what_main_charged(#[case] name: &str) {
    let case = fixture_case(FIXTURE, name);
    let model = case["model"].as_str().unwrap();
    let source = TestSource::with_row(model, units_row(&case["row"]));
    let query = VideoQuery {
        header: header("openai", model),
        seconds: case["seconds"].as_f64(),
        resolution: case["resolution"]
            .as_str()
            .map(|resolution| resolution.to_lowercase().into()),
        videos: 1,
        input_images: 0,
        input_image_rows: Vec::new(),
    };
    let total = match price(&source, &query) {
        PriceOutcome::Priced(receipt) => receipt.total,
        PriceOutcome::Unpriced(_) => 0.0,
    };
    assert_eq!(Some(total), case["expected"].as_f64());
}
