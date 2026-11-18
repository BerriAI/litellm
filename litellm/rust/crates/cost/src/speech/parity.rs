//! Parity with the speech branch of main's `cost_per_token`. `fixtures/speech_parity.json` was captured once from
//! main's Python: each case holds the cost fields `get_model_info` returned (or the Lyria row), the billed characters
//! and main's `(prompt, completion)`, or that it raised. The adapter raises on any unpriced speech call, as main did

use rstest::rstest;
use serde_json::Value;

use super::{SpeechQuery, price};
use crate::query::{Header, PriceSource};
use crate::receipt::PriceOutcome;
use crate::testing::{TestSource, expected_sides, fixture_case, header, row_ref, units_row};

const FIXTURE: &str = include_str!("../../fixtures/speech_parity.json");
const LYRIA: &str = "lyria";

fn case(name: &str) -> Value {
    fixture_case(FIXTURE, name)
}

#[rstest]
#[case::tts_1("tts_1")]
#[case::azure_tts("azure_tts")]
#[case::elevenlabs("elevenlabs")]
#[case::dashscope_characters("dashscope_characters")]
#[case::uncounted_characters("uncounted_characters")]
#[case::lyria_generation("lyria_generation")]
fn a_speech_call_costs_what_main_charged(#[case] name: &str) {
    let case = case(name);
    let model = case["model"].as_str().unwrap();
    let base = header(case["provider"].as_str().unwrap(), model);
    let query = match case["lyria"].is_null() {
        true => SpeechQuery {
            header: base,
            characters: case["characters"].as_u64(),
            generations: None,
        },
        false => SpeechQuery {
            header: Header {
                source: PriceSource::Rows {
                    rows: vec![row_ref(LYRIA)],
                },
                ..base
            },
            characters: None,
            generations: Some(1),
        },
    };
    let source = TestSource::with_row(model, units_row(&case["row"]))
        .and_row(LYRIA, units_row(&case["lyria"]));
    let sides = match price(&source, &query) {
        PriceOutcome::Priced(receipt) => Some((receipt.prompt.get(), receipt.completion.get())),
        PriceOutcome::Unpriced(_) => None,
    };
    assert_eq!(sides, expected_sides(&case));
}
