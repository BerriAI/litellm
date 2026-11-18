//! What `price` answers. A priced call carries its receipt: the meter-tagged lines, the rates they were billed at,
//! each adjustment, and the breakdown values main stores on the logging object. An unpriced call says why and is
//! never 0.0

use crate::adjust::{Adjusted, AdjustmentKey, Discount, Margin, adjust};
use crate::money::Usd;
use crate::price::Meter;
use crate::priced::Line;
use crate::query::{AdjustmentPolicy, Basis, Candidate, Kind};
use crate::row::{RowRef, RowSource};

#[derive(Clone, Debug, PartialEq)]
pub enum PriceOutcome {
    Priced(Box<Receipt>),
    Unpriced(Unpriced),
}

#[derive(Clone, Debug, PartialEq)]
pub enum Unpriced {
    /// No candidate has a row
    NoRow { tried: Vec<Candidate> },
    /// The row prices none of the meters the call used
    MissingRates { row: RowRef, missing: Vec<Meter> },
    /// The row or price source bills the call by a meter whose quantity the call did not report. `row` is `None` for
    /// a price the caller configured rather than a row
    MissingUsage { row: Option<RowRef>, meter: Meter },
    /// The rows given do not price the call, and the caller has not fetched the catalog row it deferred
    CatalogDeferred,
    /// This crate does not price this kind, or this case of it, yet. The caller prices it the old way
    NotPorted(Kind),
}

#[derive(Clone, Debug, PartialEq)]
pub struct Receipt {
    /// The candidate that priced, `None` for custom pricing
    pub candidate: Option<Candidate>,
    /// The row that priced, `None` for custom pricing
    pub row: Option<RowRef>,
    /// One line per billed meter, uplifts included
    pub lines: Vec<Line>,
    /// Meters with usage the row has no rate for, which main bills at 0.0
    pub defaulted_to_zero: Vec<Meter>,
    /// The per-token rates the token-type breakdown reads, after tiers, off-peak and uplifts. `None` when the row
    /// main reads them from does not exist, and the breakdown is then all zero
    pub rates: Option<BilledRates>,
    /// The prompt-side lines summed in main's order, then uplifted
    pub prompt: Usd,
    /// The completion-side lines summed in main's order, then uplifted
    pub completion: Usd,
    /// Built-in tool fees, such as web search
    pub tools: Usd,
    /// Provider fees beyond tokens and tools, such as the Azure AI Model Router fee
    pub additional: Usd,
    pub discount: Discount,
    pub margin: Margin,
    pub breakdown: Breakdown,
    /// The rows or units a call billed as separately priced parts, such as each guardrail counter. A part no price
    /// covers stays `Unpriced` beside the others, and the totals count only the priced ones
    pub parts: Vec<Part>,
    /// Read discount minus write premium, main's `calculate_prompt_caching_savings`. `None` for custom pricing
    pub cache_savings: Option<f64>,
    pub basis: Basis,
    /// A cost the provider reported, which stands for the priced lines as the cost before discount and margin
    pub reported: Option<Usd>,
    /// The cost after discount and margin
    pub total: f64,
}

/// One separately priced part of a call
#[derive(Clone, Debug, PartialEq)]
pub struct Part {
    /// What the part bills, such as a guardrail counter name
    pub label: Box<str>,
    pub outcome: PartOutcome,
}

#[derive(Clone, Debug, PartialEq)]
pub enum PartOutcome {
    Priced {
        row: Option<RowRef>,
        amount: Usd,
    },
    /// No price covers this part. It adds nothing to the total and is never read as free
    Unpriced,
}

impl Part {
    pub fn amount(&self) -> Option<Usd> {
        match self.outcome {
            PartOutcome::Priced { amount, .. } => Some(amount),
            PartOutcome::Unpriced => None,
        }
    }
}

/// `BilledTokenRates`
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct BilledRates {
    pub input: f64,
    pub output: f64,
    pub cache_read: f64,
    pub cache_read_audio: f64,
    /// `None` when the row has no cached image rate of its own
    pub cache_read_image: Option<f64>,
    pub cache_write: f64,
    pub cache_write_one_hour: f64,
    pub cache_write_twelve_hours: f64,
    pub reasoning: f64,
}

/// The token-type breakdown main stores beside the totals
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Breakdown {
    pub cache_read: f64,
    pub cache_creation: f64,
    pub reasoning: f64,
}

impl Receipt {
    /// The token cost before tools, fees and adjustments
    pub fn token_subtotal(&self) -> Usd {
        self.prompt + self.completion
    }

    /// The cost before discount and margin, main's `original_cost`
    pub fn original(&self) -> f64 {
        self.reported
            .unwrap_or(self.token_subtotal() + self.tools + self.additional)
            .get()
    }

    /// Applies the discount and margin `policy` asks for and settles the total. The discount is keyed on `model`
    /// unless the policy names another
    pub(crate) fn settled(
        self,
        source: &impl RowSource,
        policy: &AdjustmentPolicy,
        model: Option<&str>,
    ) -> Self {
        let original = self.original();
        let Adjusted {
            cost,
            discount,
            margin,
        } = match policy {
            AdjustmentPolicy::None => Adjusted {
                cost: original,
                discount: Discount::default(),
                margin: Margin::default(),
            },
            AdjustmentPolicy::Apply {
                provider,
                model: keyed,
                region,
            } => adjust(
                source.discounts(),
                source.margins(),
                original,
                &AdjustmentKey {
                    provider: provider.as_deref(),
                    model: keyed.as_deref().or(model),
                    region: region.as_deref(),
                },
            ),
        };
        Self {
            discount,
            margin,
            total: cost,
            ..self
        }
    }
}

#[cfg(feature = "test-support")]
impl Receipt {
    /// A receipt for `total` on the prompt alone, billed at the Unix epoch, for tests of what carries a receipt
    pub fn of_total(total: f64) -> Self {
        Self {
            candidate: None,
            row: None,
            lines: Vec::new(),
            defaulted_to_zero: Vec::new(),
            rates: None,
            prompt: Usd::new(total).unwrap_or_default(),
            completion: Usd::ZERO,
            tools: Usd::ZERO,
            additional: Usd::ZERO,
            discount: Discount::default(),
            margin: Margin::default(),
            breakdown: Breakdown::default(),
            parts: Vec::new(),
            cache_savings: None,
            reported: None,
            basis: Basis {
                service_tier: crate::price::ServiceTier::default(),
                data_residency: None,
                vertex_location: None,
                at: jiff::Timestamp::UNIX_EPOCH,
            },
            total,
        }
    }
}

impl From<Unpriced> for PriceOutcome {
    fn from(unpriced: Unpriced) -> Self {
        Self::Unpriced(unpriced)
    }
}

impl From<Receipt> for PriceOutcome {
    fn from(receipt: Receipt) -> Self {
        Self::Priced(Box::new(receipt))
    }
}
