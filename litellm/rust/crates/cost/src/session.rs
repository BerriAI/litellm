//! The `session` kind, a port of `handle_realtime_stream_cost_calculation` (`cost_calculator.py`): a realtime
//! session's tokens and its Live `response.done` turns on the receipt's prompt and completion, and Live voice seconds,
//! xAI audio seconds and text messages, translation seconds and transcription clips as one part each. A part is priced
//! from the first of its rows, in the order main tries the session's model names, that lists its rate. A part no row
//! prices is `Unpriced`, which main billed as 0.0. Main applies no discount or margin to a session

use crate::money::{UnitPrice, Usd};
use crate::price::{Channel, InputUnit, Meter, OutputUnit, ServiceTier, TokenKind};
use crate::priced::{Line, Quantity, Side};
use crate::query::{
    AdjustmentPolicy, Basis, Candidate, Header, PriceSource, TokenRules, TokenUsage, TokensQuery,
};
use crate::receipt::{Part, PartOutcome, PriceOutcome, Receipt};
use crate::row::{RowRef, RowSource};
use crate::units::{self, Billed};

const SECOND: Meter = Meter::Input(InputUnit::Second);
const OUTPUT_SECOND: Meter = Meter::Output(OutputUnit::Second(None));
const TEXT_MESSAGE: Meter = Meter::Input(InputUnit::TextMessage);
const AUDIO_TOKEN: Meter = Meter::Input(InputUnit::Token(TokenKind::Audio));

/// What a realtime session used beside its tokens. Each part lists its rows in the order main tries them
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize))]
pub struct SessionQuery {
    #[cfg_attr(feature = "serde", serde(flatten))]
    pub header: Header,
    /// Each `conversation.item.input_audio_transcription.completed` clip
    #[cfg_attr(feature = "serde", serde(default))]
    pub transcription: Vec<TranscriptionClip>,
    #[cfg_attr(feature = "serde", serde(default))]
    pub translation: Option<Translation>,
    /// Live voice seconds at `input_cost_per_second`
    #[cfg_attr(feature = "serde", serde(default))]
    pub live_voice: Option<Metered>,
    /// xAI audio seconds at `input_cost_per_second`
    #[cfg_attr(feature = "serde", serde(default))]
    pub audio_duration: Option<Metered>,
    /// xAI text messages at `input_cost_per_text_message`
    #[cfg_attr(feature = "serde", serde(default))]
    pub text_messages: Option<Metered>,
    /// The session's combined token usage
    #[cfg_attr(feature = "serde", serde(default))]
    pub tokens: Option<SessionTokens>,
    /// Each distinct Live `response.done` turn, priced on the model it names
    #[cfg_attr(feature = "serde", serde(default))]
    pub turns: Vec<SessionTurn>,
}

/// The session's tokens, billed by `generic_cost_per_token` at the first candidate that bills them something or whose
/// cost map entry declares a price, so a deployment priced at zero still wins over the public rates. A candidate
/// whose pricing fails is skipped, and none accepted bills nothing
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize))]
pub struct SessionTokens {
    pub usage: TokenUsage,
    pub candidates: Vec<TokenCandidate>,
}

#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize))]
pub struct TokenCandidate {
    pub candidate: Candidate,
    pub row: RowRef,
    /// The candidate's cost map entry sets a rate of its own, even a zero one
    pub declares: bool,
}

/// One Live `response.done` turn, billed nothing when its model has no row or its pricing fails
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize))]
pub struct SessionTurn {
    pub candidate: Candidate,
    pub usage: TokenUsage,
    pub row: Option<RowRef>,
}

#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize))]
pub struct Metered {
    pub quantity: f64,
    pub rows: Vec<RowRef>,
}

/// A translation session's closed seconds, each direction at the first row that declares its per-second rate
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize))]
pub struct Translation {
    pub input_seconds: f64,
    pub output_seconds: f64,
    pub rows: Vec<RowRef>,
}

/// One transcribed clip, billed by its ASR model: the session's pricing override first, then the clip's model
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize))]
pub struct TranscriptionClip {
    pub rows: Vec<RowRef>,
    pub usage: ClipUsage,
}

#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Deserialize),
    serde(tag = "type", rename_all = "snake_case")
)]
pub enum ClipUsage {
    Duration {
        seconds: f64,
    },
    Tokens {
        audio: f64,
        text: f64,
        output: f64,
    },
    /// A usage type main does not bill
    Other,
}

/// The parts in the order main adds them to the session's cost
const PARTS: [&str; 5] = [
    "transcription_cost",
    "translation_cost",
    "live_voice_cost",
    "audio_duration_cost",
    "text_message_cost",
];

pub(crate) fn price<S: RowSource>(source: &S, query: &SessionQuery) -> PriceOutcome {
    let rate = |rows: &[RowRef], meters: &[Meter]| -> Option<f64> {
        rows.iter().find_map(|at| {
            let row = source.row(at)?;
            meters
                .iter()
                .find_map(|meter| units::listed(row, meter, Channel::Online))
                .map(UnitPrice::get)
        })
    };
    let metered = |used: &Option<Metered>, meter: Meter| -> Option<Bill> {
        let used = used.as_ref()?;
        rate(&used.rows, std::slice::from_ref(&meter))
            .map(|rate| Bill::of(meter, used.quantity, rate))
    };
    let bills = [
        transcription(&query.transcription, &rate),
        translation(query.translation.as_ref(), &rate),
        metered(&query.live_voice, SECOND),
        metered(&query.audio_duration, SECOND),
        metered(&query.text_messages, TEXT_MESSAGE),
    ];
    let parts = PARTS
        .iter()
        .zip(&bills)
        .map(|(label, bill)| Part {
            label: (*label).into(),
            outcome: match bill {
                Some(bill) => PartOutcome::Priced {
                    row: None,
                    amount: Usd::new(bill.amount).unwrap_or_default(),
                },
                None => PartOutcome::Unpriced,
            },
        })
        .collect();
    let token_sides = |candidate: &Candidate, row: &RowRef, usage: &TokenUsage| {
        token_receipt(source, &query.header.basis, candidate, row, usage)
    };
    let session_tokens = query
        .tokens
        .as_ref()
        .and_then(|tokens| {
            tokens.candidates.iter().find_map(|offered| {
                let (prompt, completion) =
                    token_sides(&offered.candidate, &offered.row, &tokens.usage)?;
                (prompt + completion > 0.0 || offered.declares).then_some((prompt, completion))
            })
        })
        .unwrap_or((0.0, 0.0));
    let turns: Vec<(f64, f64)> = query
        .turns
        .iter()
        .map(|turn| {
            turn.row
                .as_ref()
                .and_then(|row| token_sides(&turn.candidate, row, &turn.usage))
                .unwrap_or((0.0, 0.0))
        })
        .collect();
    let prompt = session_tokens.0 + turns.iter().fold(0.0, |sum, turn| sum + turn.0);
    let completion = session_tokens.1 + turns.iter().fold(0.0, |sum, turn| sum + turn.1);
    let total = bills.iter().fold(prompt + completion, |sum, bill| {
        sum + bill.as_ref().map_or(0.0, |bill| bill.amount)
    });
    let lines = bills.into_iter().flatten().map(Bill::line).collect();
    Receipt {
        parts,
        prompt: Usd::new(prompt).unwrap_or_default(),
        completion: Usd::new(completion).unwrap_or_default(),
        total,
        ..units::receipt(source, &query.header, None, Billed::of(lines, Vec::new()))
    }
    .into()
}

/// `generic_cost_per_token` of `usage` on `row` with the session's data residency, `None` where main's call raised
fn token_receipt<S: RowSource>(
    source: &S,
    basis: &Basis,
    candidate: &Candidate,
    row: &RowRef,
    usage: &TokenUsage,
) -> Option<(f64, f64)> {
    let query = TokensQuery {
        rules: TokenRules::Generic,
        ..TokensQuery::new(
            Header {
                candidates: vec![candidate.clone()],
                source: PriceSource::Row { row: row.clone() },
                basis: Basis {
                    service_tier: ServiceTier::Standard,
                    vertex_location: None,
                    ..basis.clone()
                },
                adjustments: AdjustmentPolicy::None,
                reported: None,
            },
            usage.clone(),
        )
    };
    match crate::tokens::price(source, &query) {
        PriceOutcome::Priced(receipt) => Some((receipt.prompt.get(), receipt.completion.get())),
        PriceOutcome::Unpriced(_) => None,
    }
}

/// A part's amount as main computed it and the meter and rate it shows as a line
struct Bill {
    meter: Meter,
    quantity: f64,
    rate: f64,
    amount: f64,
}

impl Bill {
    fn of(meter: Meter, quantity: f64, rate: f64) -> Self {
        Self {
            meter,
            quantity,
            rate,
            amount: quantity * rate,
        }
    }

    fn line(self) -> Line {
        Line {
            meter: self.meter,
            quantity: Quantity::Measured(self.quantity),
            unit_price: UnitPrice::new(self.rate)
                .unwrap_or(UnitPrice::new(0.0).expect("zero is a price")),
            amount: Usd::new(self.amount).unwrap_or_default(),
            fallback_from: None,
            side: Side::Prompt,
        }
    }
}

/// `handle_realtime_transcription_cost_calculation`: every clip summed, a rate no row declares billed at 0.0
fn transcription(
    clips: &[TranscriptionClip],
    rate: &impl Fn(&[RowRef], &[Meter]) -> Option<f64>,
) -> Option<Bill> {
    let priced: Vec<(f64, bool)> = clips
        .iter()
        .map(|clip| {
            let rate = |meters: &[Meter]| rate(&clip.rows, meters);
            match clip.usage {
                ClipUsage::Duration { seconds } => {
                    let found = rate(&[SECOND]);
                    (seconds * found.unwrap_or(0.0), found.is_some())
                }
                ClipUsage::Tokens {
                    audio,
                    text,
                    output,
                } => {
                    let rates = [
                        rate(&[AUDIO_TOKEN, Meter::TEXT_INPUT]),
                        rate(&[Meter::TEXT_INPUT]),
                        rate(&[Meter::TEXT_OUTPUT]),
                    ];
                    let [audio_rate, text_rate, output_rate] =
                        rates.map(|found| found.unwrap_or(0.0));
                    (
                        audio * audio_rate + text * text_rate + output * output_rate,
                        rates.iter().any(Option::is_some),
                    )
                }
                ClipUsage::Other => (0.0, false),
            }
        })
        .collect();
    priced.iter().any(|(_, found)| *found).then(|| Bill {
        meter: SECOND,
        quantity: clips.len() as f64,
        rate: 0.0,
        amount: priced.iter().fold(0.0, |sum, (cost, _)| sum + cost),
    })
}

/// `handle_realtime_translation_cost_calculation`: each direction at its own first declared rate
fn translation(
    used: Option<&Translation>,
    rate: &impl Fn(&[RowRef], &[Meter]) -> Option<f64>,
) -> Option<Bill> {
    let used = used.filter(|used| used.input_seconds > 0.0 || used.output_seconds > 0.0)?;
    let input = rate(&used.rows, &[SECOND]);
    let output = rate(&used.rows, &[OUTPUT_SECOND]);
    (input.is_some() || output.is_some()).then(|| Bill {
        meter: SECOND,
        quantity: used.input_seconds,
        rate: input.unwrap_or(0.0),
        amount: used.input_seconds * input.unwrap_or(0.0)
            + used.output_seconds * output.unwrap_or(0.0),
    })
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;
    use crate::price::ServiceTier;
    use crate::testing::{BASE, TestRow, TestSource, header, priced, prices, row_ref};

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

    fn source() -> TestSource {
        TestSource::with_row("override", row(&[(SECOND, 0.002)]))
            .and_row(
                "asr",
                row(&[
                    (SECOND, 0.001),
                    (Meter::TEXT_INPUT, 1e-6),
                    (Meter::TEXT_OUTPUT, 4e-6),
                ]),
            )
            .and_row(
                "audio-asr",
                row(&[(AUDIO_TOKEN, 3e-6), (Meter::TEXT_INPUT, 1e-6)]),
            )
            .and_row("unpriced", TestRow::default())
            .and_row("translator", row(&[(SECOND, 0.01), (OUTPUT_SECOND, 0.02)]))
            .and_row("messages", row(&[(TEXT_MESSAGE, 0.0001)]))
    }

    fn rows(keys: &[&str]) -> Vec<RowRef> {
        keys.iter().map(|key| row_ref(key)).collect()
    }

    fn session(query: SessionQuery) -> Vec<(String, Option<f64>)> {
        priced(price(&source(), &query))
            .parts
            .iter()
            .map(|part| (part.label.to_string(), part.amount().map(Usd::get)))
            .collect()
    }

    fn empty() -> SessionQuery {
        SessionQuery {
            header: header("openai", "gpt-realtime"),
            transcription: Vec::new(),
            translation: None,
            live_voice: None,
            audio_duration: None,
            text_messages: None,
            tokens: None,
            turns: Vec::new(),
        }
    }

    fn part(parts: &[(String, Option<f64>)], label: &str) -> Option<f64> {
        parts
            .iter()
            .find(|(name, _)| name == label)
            .and_then(|(_, amount)| *amount)
    }

    #[rstest]
    #[case::first_priced_row_wins(&["unpriced", "override", "asr"], Some(30.0 * 0.002))]
    #[case::a_row_without_the_rate_is_skipped(&["unpriced", "asr"], Some(30.0 * 0.001))]
    #[case::no_row_prices_it(&["unpriced"], None)]
    fn live_voice_bills_seconds_at_the_first_listed_rate(
        #[case] keys: &[&str],
        #[case] expected: Option<f64>,
    ) {
        let parts = session(SessionQuery {
            live_voice: Some(Metered {
                quantity: 30.0,
                rows: rows(keys),
            }),
            ..empty()
        });
        assert_eq!(part(&parts, "live_voice_cost"), expected);
    }

    #[rstest]
    fn xai_audio_and_text_messages_are_separate_parts() {
        let parts = session(SessionQuery {
            audio_duration: Some(Metered {
                quantity: 12.5,
                rows: rows(&["asr"]),
            }),
            text_messages: Some(Metered {
                quantity: 7.0,
                rows: rows(&["asr", "messages"]),
            }),
            ..empty()
        });
        assert_eq!(part(&parts, "audio_duration_cost"), Some(12.5 * 0.001));
        assert_eq!(part(&parts, "text_message_cost"), Some(7.0 * 0.0001));
        assert_eq!(part(&parts, "live_voice_cost"), None);
    }

    #[rstest]
    fn transcription_clips_sum_by_usage_type_with_the_override_first() {
        let clips = vec![
            TranscriptionClip {
                rows: rows(&["override", "asr"]),
                usage: ClipUsage::Duration { seconds: 4.0 },
            },
            TranscriptionClip {
                rows: rows(&["audio-asr"]),
                usage: ClipUsage::Tokens {
                    audio: 100.0,
                    text: 20.0,
                    output: 50.0,
                },
            },
            TranscriptionClip {
                rows: rows(&["asr"]),
                usage: ClipUsage::Other,
            },
        ];
        let parts = session(SessionQuery {
            transcription: clips,
            ..empty()
        });
        let tokens = 100.0 * 3e-6 + 20.0 * 1e-6 + 50.0 * 0.0;
        assert_eq!(
            part(&parts, "transcription_cost"),
            Some(0.0 + 4.0 * 0.002 + tokens + 0.0)
        );
    }

    #[rstest]
    #[case::both_directions(10.0, 5.0, Some(10.0 * 0.01 + 5.0 * 0.02))]
    #[case::no_seconds(0.0, 0.0, None)]
    fn translation_bills_each_direction_at_its_rate(
        #[case] input_seconds: f64,
        #[case] output_seconds: f64,
        #[case] expected: Option<f64>,
    ) {
        let parts = session(SessionQuery {
            translation: Some(Translation {
                input_seconds,
                output_seconds,
                rows: rows(&["unpriced", "translator"]),
            }),
            ..empty()
        });
        assert_eq!(part(&parts, "translation_cost"), expected);
    }

    #[rstest]
    fn the_total_sums_the_priced_parts_in_mains_order() {
        let receipt = priced(price(
            &source(),
            &SessionQuery {
                live_voice: Some(Metered {
                    quantity: 30.0,
                    rows: rows(&["asr"]),
                }),
                text_messages: Some(Metered {
                    quantity: 7.0,
                    rows: rows(&["messages"]),
                }),
                ..empty()
            },
        ));
        assert_eq!(receipt.total, 30.0 * 0.001 + 7.0 * 0.0001);
    }

    fn tokens(prompt: u64, completion: u64) -> TokenUsage {
        crate::testing::usage(prompt, completion)
    }

    fn openai(key: &str) -> Candidate {
        crate::testing::candidate("openai".parse().unwrap(), key)
    }

    fn offered(key: &str, declares: bool) -> TokenCandidate {
        TokenCandidate {
            candidate: openai(key),
            row: row_ref(key),
            declares,
        }
    }

    fn token_source() -> TestSource {
        source()
            .and_row(
                "free",
                row(&[(Meter::TEXT_INPUT, 0.0), (Meter::TEXT_OUTPUT, 0.0)]),
            )
            .and_row(
                "public",
                row(&[(Meter::TEXT_INPUT, 1.0), (Meter::TEXT_OUTPUT, 2.0)]),
            )
            .and_row(
                "turn",
                row(&[(Meter::TEXT_INPUT, 3.0), (Meter::TEXT_OUTPUT, 5.0)]),
            )
    }

    #[rstest]
    #[case::a_free_row_that_declares_wins(true, (0.0, 0.0))]
    #[case::a_free_row_that_declares_nothing_falls_through(false, (10.0, 4.0))]
    fn session_tokens_bill_the_first_accepted_candidate(
        #[case] declares: bool,
        #[case] expected: (f64, f64),
    ) {
        let receipt = priced(price(
            &token_source(),
            &SessionQuery {
                tokens: Some(SessionTokens {
                    usage: tokens(10, 2),
                    candidates: vec![offered("free", declares), offered("public", false)],
                }),
                ..empty()
            },
        ));
        assert_eq!((receipt.prompt.get(), receipt.completion.get()), expected);
    }

    #[rstest]
    fn turns_add_to_the_session_tokens_and_the_parts_to_the_total() {
        let turn = |row: Option<&str>| SessionTurn {
            candidate: openai("turn"),
            usage: tokens(1, 1),
            row: row.map(row_ref),
        };
        let receipt = priced(price(
            &token_source(),
            &SessionQuery {
                tokens: Some(SessionTokens {
                    usage: tokens(10, 2),
                    candidates: vec![offered("public", false)],
                }),
                turns: vec![turn(Some("turn")), turn(None), turn(Some("turn"))],
                live_voice: Some(Metered {
                    quantity: 30.0,
                    rows: rows(&["asr"]),
                }),
                ..empty()
            },
        ));
        assert_eq!(receipt.prompt.get(), 10.0 + 3.0 + 3.0);
        assert_eq!(receipt.completion.get(), 4.0 + 5.0 + 5.0);
        assert_eq!(receipt.total, 16.0 + 14.0 + 30.0 * 0.001);
    }
}
