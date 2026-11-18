//! Parity with the transcription branch of main's `cost_per_token` and `transcription_cost`.
//! `fixtures/transcription_parity.json` was captured once from main's Python: each case holds the cost fields of the
//! cost map entry, the billed seconds, whether the usage reported tokens, and main's `(prompt, completion)`. The
//! adapter answers a call no row prices with `(0.0, 0.0)`, as main did

use rstest::rstest;
use serde_json::Value;

use super::{TranscriptionQuery, price};
use crate::receipt::PriceOutcome;
use crate::testing::{TestSource, expected_sides, fixture_case, header, units_row};

const FIXTURE: &str = include_str!("../../fixtures/transcription_parity.json");

fn case(name: &str) -> Value {
    fixture_case(FIXTURE, name)
}

#[rstest]
#[case::whisper_seconds("whisper_seconds")]
#[case::azure_whisper_output_rate("azure_whisper_output_rate")]
#[case::deepgram_input_rate("deepgram_input_rate")]
#[case::assemblyai_zero_output_rate("assemblyai_zero_output_rate")]
#[case::second_only_row_with_tokens("second_only_row_with_tokens")]
#[case::azure_stt("azure_stt")]
fn a_transcription_costs_what_main_charged(#[case] name: &str) {
    let case = case(name);
    let model = case["model"].as_str().unwrap();
    let source = TestSource::with_row(model, units_row(&case["row"]));
    let query = TranscriptionQuery {
        header: header(case["provider"].as_str().unwrap(), model),
        seconds: case["seconds"].as_f64().unwrap(),
        tokens: case["tokens"].as_bool().unwrap(),
    };
    let sides = match price(&source, &query) {
        PriceOutcome::Priced(receipt) => (receipt.prompt.get(), receipt.completion.get()),
        PriceOutcome::Unpriced(_) => (0.0, 0.0),
    };
    assert_eq!(Some(sides), expected_sides(&case));
}
