use std::time::Duration;

use crate::money::{UnitPrice, Usd};
use crate::price::Meter;

#[derive(Clone, Debug, PartialEq)]
pub enum Outcome {
    Priced(Priced),
    /// No meter with usage had a price
    Unpriced {
        missing: Vec<Meter>,
    },
}

#[derive(Clone, Debug, PartialEq)]
pub struct Priced {
    /// One line per billed meter, fallbacks included
    pub lines: Vec<Line>,
    /// Meters with usage and no price and no fallback
    pub missing: Vec<Meter>,
    /// Sum of `lines`
    pub total: Usd,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Line {
    /// The meter this line bills
    pub meter: Meter,
    /// How much of the meter the call used
    pub quantity: Quantity,
    /// Rate after off-peak and before uplifts
    pub unit_price: UnitPrice,
    /// `unit_price * quantity` times every uplift that applies
    pub amount: Usd,
    /// The meter whose rate was borrowed, when this meter had none
    pub fallback_from: Option<Meter>,
    /// Which half of main's `(prompt, completion)` pair the line belongs to
    pub side: Side,
}

/// A billed amount of one meter
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Quantity {
    Count(u64),
    Duration(Duration),
}

/// Meters main bills as one total (video, Replicate, Amazon Transcribe) take a fixed side, chosen in 09
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Side {
    Prompt,
    Completion,
}
