//! The `transcription` kind, a port of the transcription branch of `cost_per_token` (`cost_calculator.py`) and of
//! `cost_per_second` (`llms/openai/cost_calculation.py`): the file's seconds at a positive `output_cost_per_second`
//! on the completion side, else at `input_cost_per_second` on the prompt side. A call whose usage reports tokens on a
//! row that declares token rates, or declares no per-second rate, bills its tokens through `generic_cost_per_token`

use crate::price::{Channel, InputUnit, Meter, OutputUnit, TokenKind};
use crate::query::Header;
use crate::receipt::{PriceOutcome, Unpriced};
use crate::row::RowSource;
use crate::units::{self, TokenBilled};
use crate::usage::{Media, Usage};

const INPUT_SECOND: Meter = Meter::Input(InputUnit::Second);
const OUTPUT_SECOND: Meter = Meter::Output(OutputUnit::Second(None));

/// One transcription call. The header's row is the cost map entry as written, since main reads which rates it
/// declares off the raw entry
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize))]
pub struct TranscriptionQuery {
    #[cfg_attr(feature = "serde", serde(flatten))]
    pub header: Header,
    /// The billed length of the audio
    pub seconds: f64,
    /// The token usage, when the response reported prompt or completion tokens
    #[cfg_attr(feature = "serde", serde(default))]
    pub tokens: Option<TokenBilled>,
}

pub(crate) fn price<S: RowSource>(source: &S, query: &TranscriptionQuery) -> PriceOutcome {
    let Some(named) = units::named(source, &query.header) else {
        return units::no_row(&query.header);
    };
    let listed = |meter: &Meter| units::listed(named.row, meter, Channel::Online);
    let token_meters = [
        Meter::TEXT_INPUT,
        Meter::TEXT_OUTPUT,
        Meter::Input(InputUnit::Token(TokenKind::Audio)),
        Meter::Output(OutputUnit::Token(TokenKind::Audio)),
    ];
    let declares_tokens = token_meters.iter().any(|meter| listed(meter).is_some());
    let declares_seconds = listed(&INPUT_SECOND).is_some() || listed(&OUTPUT_SECOND).is_some();
    if let Some(tokens) = query
        .tokens
        .as_ref()
        .filter(|_| declares_tokens || !declares_seconds)
    {
        return units::generic_tokens(source, &query.header, tokens);
    }
    let media = match (
        listed(&OUTPUT_SECOND).filter(|rate| rate.get() > 0.0),
        listed(&INPUT_SECOND),
    ) {
        (Some(_), _) => Media {
            output_seconds: query.seconds,
            ..Media::default()
        },
        (None, Some(_)) => Media {
            input_seconds: query.seconds,
            ..Media::default()
        },
        (None, None) => {
            return Unpriced::MissingRates {
                row: named.at,
                missing: vec![INPUT_SECOND],
            }
            .into();
        }
    };
    let usage = Usage {
        media,
        ..Usage::default()
    };
    units::bill(
        source,
        &query.header,
        &named,
        &usage,
        &units::call(&query.header.basis),
    )
}

#[cfg(test)]
mod parity {
    //! Parity with the transcription branch of main's `cost_per_token` and `transcription_cost`.
    //! `fixtures/transcription_parity.json` was captured once from main's Python: each case holds the cost fields of the
    //! cost map entry, the billed seconds, whether the usage reported tokens, and main's `(prompt, completion)`. The
    //! adapter answers a call no row prices with `(0.0, 0.0)`, as main did

    use rstest::rstest;
    use serde_json::Value;

    use super::{TranscriptionQuery, price};
    use crate::receipt::PriceOutcome;
    use crate::testing::{TestSource, expected_sides, fixture_case, header, row_ref, units_row};
    use crate::units::TokenBilled;

    const FIXTURE: &str = include_str!("../fixtures/transcription_parity.json");

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
            tokens: case["tokens"].as_bool().unwrap().then(|| TokenBilled {
                row: row_ref(model),
                usage: Default::default(),
            }),
        };
        let sides = match price(&source, &query) {
            PriceOutcome::Priced(receipt) => (receipt.prompt.get(), receipt.completion.get()),
            PriceOutcome::Unpriced(_) => (0.0, 0.0),
        };
        assert_eq!(Some(sides), expected_sides(&case));
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;
    use crate::price::ServiceTier;
    use crate::testing::{BASE, TestRow, TestSource, header, priced, prices, row_ref, usage};

    fn row(rates: &[(Meter, f64)]) -> TestRow {
        TestRow {
            prices: prices(
                &rates
                    .iter()
                    .map(|(meter, rate)| (meter.clone(), ServiceTier::Standard, BASE, *rate))
                    .collect::<Vec<_>>(),
            ),
            ..TestRow::default()
        }
    }

    fn query(tokens: bool) -> TranscriptionQuery {
        TranscriptionQuery {
            header: header("openai", "whisper-1"),
            seconds: 61.37,
            tokens: tokens.then(|| TokenBilled {
                row: row_ref("tokens-row"),
                usage: usage(1_000, 200),
            }),
        }
    }

    fn sides(source: &TestSource, tokens: bool) -> (f64, f64) {
        let receipt = priced(price(source, &query(tokens)));
        (receipt.prompt.get(), receipt.completion.get())
    }

    #[rstest]
    #[case::input_rate(row(&[(INPUT_SECOND, 0.0001)]), (0.0001 * 61.37, 0.0))]
    #[case::output_rate(row(&[(OUTPUT_SECOND, 0.0002)]), (0.0, 0.0002 * 61.37))]
    #[case::output_ahead_of_input(
        row(&[(INPUT_SECOND, 0.0001), (OUTPUT_SECOND, 0.0002)]),
        (0.0, 0.0002 * 61.37)
    )]
    #[case::zero_output_rate_reads_input(
        row(&[(INPUT_SECOND, 0.0001), (OUTPUT_SECOND, 0.0)]),
        (0.0001 * 61.37, 0.0)
    )]
    fn the_file_bills_by_the_second(#[case] priced_row: TestRow, #[case] expected: (f64, f64)) {
        assert_eq!(
            sides(&TestSource::with_row("whisper-1", priced_row), false),
            expected
        );
    }

    #[rstest]
    fn a_second_only_row_bills_seconds_even_when_tokens_are_reported() {
        let source = TestSource::with_row("whisper-1", row(&[(INPUT_SECOND, 0.0001)]));
        assert_eq!(sides(&source, true), (0.0001 * 61.37, 0.0));
    }

    /// The tokens bill on the row `generic_cost_per_token` read, not the declared entry, whose rates decide only whether
    /// tokens bill at all
    #[rstest]
    #[case::token_row(row(&[(INPUT_SECOND, 0.0001), (Meter::TEXT_INPUT, 9.0)]))]
    #[case::no_second_rate(row(&[]))]
    fn reported_tokens_on_a_row_main_bills_by_token_bill_by_token(#[case] declared: TestRow) {
        let source = TestSource::with_row("whisper-1", declared).and_row(
            "tokens-row",
            row(&[(Meter::TEXT_INPUT, 1e-6), (Meter::TEXT_OUTPUT, 4e-6)]),
        );
        assert_eq!(sides(&source, true), (1e-6 * 1_000.0, 4e-6 * 200.0));
    }

    #[rstest]
    fn a_row_without_a_second_rate_is_missing_it() {
        let source = TestSource::with_row("whisper-1", row(&[(Meter::TEXT_INPUT, 1.0)]));
        assert_eq!(
            price(&source, &query(false)),
            Unpriced::MissingRates {
                row: row_ref("whisper-1"),
                missing: vec![INPUT_SECOND]
            }
            .into()
        );
    }
}
