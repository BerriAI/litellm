//! The `session` kind, a port of what `handle_realtime_stream_cost_calculation` (`cost_calculator.py`) bills beside a
//! realtime session's tokens: Live voice seconds, xAI audio seconds and text messages, translation seconds and
//! transcription clips. Each of them is one part of the receipt, priced from the first of its rows, in the order main
//! tries the session's model names, that lists its rate. A part no row prices is `Unpriced`, which main billed as 0.0.
//! The session's tokens are priced by the token path, so the adapter adds them. Main applies no discount or margin
//! to a session

use crate::money::{UnitPrice, Usd};
use crate::price::{Channel, InputUnit, Meter, OutputUnit, TokenKind};
use crate::priced::{Line, Quantity, Side};
use crate::query::Header;
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
    let lines = bills.into_iter().flatten().map(Bill::line).collect();
    Receipt {
        parts,
        ..units::receipt(source, &query.header, None, Billed::of(lines, Vec::new()))
    }
    .into()
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
}
