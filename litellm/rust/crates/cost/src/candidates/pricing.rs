//! A token call priced over a catalog: the rows one query reads, fetched through `PricingTable` before the query runs,
//! and `priced`, the entry point every host calls with the facts `completion_cost` knows about the call. The core never
//! sees the table, so a query can only reach the rows fetched here

use std::str::FromStr;
use std::sync::Arc;

use super::{CallIdentity, CandidateTable, RowProvenance, RowRef, Tried, candidates, priced_row};
use crate::adjust::{CostConfig, Discounts, Margins};
use crate::price::ServiceTier;
use crate::query::{
    AdjustmentPolicy, Basis, Candidate, DataResidency, Header, PriceQuery, PriceSource, Provider,
    TokenUsage, TokensQuery,
};
use crate::receipt::{PriceOutcome, Receipt};
use crate::row::{LoweredRow, Provenance, RowRef as PricedRef, RowSource};

/// The key a row the caller passed for one call is priced under
const PER_CALL_ROW: &str = "requested_model_info_override";

/// What pricing asks of a catalog besides the candidate walk
pub trait PricingTable: CandidateTable {
    type Row: LoweredRow;

    /// The row under exactly `key`, lowered for billing as `provider`. `None` when the table has no such key
    fn lowered(&self, key: &str, provider: &str) -> Option<Arc<Self::Row>>;
    /// The row under exactly `key`, billed as the provider it names
    fn row_at(&self, key: &str) -> Option<RowRef>;
    fn config(&self) -> &Arc<CostConfig>;
}

/// The rows a query may read, each fetched once, and the discount and margin config of the catalog's context
#[derive(Clone, Debug)]
pub struct PricingRows<R> {
    named: Vec<(Candidate, PricedRef)>,
    rows: Vec<(PricedRef, Arc<R>)>,
    config: Arc<CostConfig>,
}

/// The rows `named` resolved to, under the candidate main priced each with, plus `rows` read by key alone. A key the
/// table no longer holds is left out, so a query on it answers as if the catalog had no row
pub fn pricing_rows<T: PricingTable + ?Sized>(
    table: &T,
    named: impl IntoIterator<Item = (Candidate, RowRef)>,
    rows: impl IntoIterator<Item = RowRef>,
) -> PricingRows<T::Row> {
    let named: Vec<(Candidate, RowRef)> = named.into_iter().collect();
    let fetched = named
        .iter()
        .map(|(_, row)| row.clone())
        .chain(rows)
        .filter_map(|row_ref| {
            let row = table.lowered(&row_ref.key, &row_ref.provider)?;
            Some((priced_ref(&row_ref), row))
        });
    PricingRows {
        rows: fetched.collect(),
        named: named
            .into_iter()
            .map(|(candidate, row)| (candidate, priced_ref(&row)))
            .collect(),
        config: Arc::clone(table.config()),
    }
}

impl<R> PricingRows<R> {
    /// Adds a row the caller passed for this call alone, under `key`
    pub fn with_per_call(self, key: &str, row: R) -> (Self, PricedRef) {
        let at = PricedRef {
            key: key.into(),
            provenance: Provenance::PerCall,
        };
        let rows = self
            .rows
            .into_iter()
            .chain([(at.clone(), Arc::new(row))])
            .collect();
        (Self { rows, ..self }, at)
    }
}

impl<R: LoweredRow> RowSource for PricingRows<R> {
    type Row = R;

    fn resolve(&self, candidate: &Candidate) -> Option<PricedRef> {
        self.named
            .iter()
            .find(|(named, _)| named == candidate)
            .map(|(_, at)| at.clone())
    }

    fn row(&self, at: &PricedRef) -> Option<&R> {
        self.rows
            .iter()
            .find(|(fetched, _)| fetched == at)
            .map(|(_, row)| &**row)
    }

    fn discounts(&self) -> &Discounts {
        &self.config.discounts
    }

    fn margins(&self) -> &Margins {
        &self.config.margins
    }
}

/// The pricing core's name for a row the candidate walk found
pub fn priced_ref(row: &RowRef) -> PricedRef {
    PricedRef {
        key: row.key.as_str().into(),
        provenance: match row.provenance {
            RowProvenance::Catalog => Provenance::Catalog,
            RowProvenance::Registered => Provenance::Registered,
        },
    }
}

/// The rows to try: the candidates main's `completion_cost` loop walks for a finished call, or the one model a
/// caller of `cost_per_token` names
#[derive(Clone, Debug)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Deserialize),
    serde(tag = "by", rename_all = "snake_case")
)]
pub enum Named {
    Call(CallIdentity),
    Model {
        model: String,
        provider: String,
        #[cfg_attr(feature = "serde", serde(default))]
        region: Option<String>,
    },
}

/// What a host knows about a finished token call: which rows to try, the usage and the conditions it was served under
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize))]
pub struct TokenCall {
    pub named: Named,
    pub usage: TokenUsage,
    #[cfg_attr(feature = "serde", serde(default))]
    pub requested_model: Option<Box<str>>,
    #[cfg_attr(feature = "serde", serde(default))]
    pub service_tier: Option<String>,
    #[cfg_attr(feature = "serde", serde(default))]
    pub data_residency: Option<String>,
    #[cfg_attr(feature = "serde", serde(default))]
    pub vertex_location: Option<Box<str>>,
    pub at: jiff::Timestamp,
    /// The discount and margin key, absent where main applies neither (`cost_per_token`)
    #[cfg_attr(feature = "serde", serde(default))]
    pub adjust: Option<AdjustFacts>,
}

#[derive(Clone, Debug, Default)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize))]
pub struct AdjustFacts {
    #[cfg_attr(feature = "serde", serde(default))]
    pub provider: Option<Box<str>>,
    #[cfg_attr(feature = "serde", serde(default))]
    pub model: Option<Box<str>>,
    #[cfg_attr(feature = "serde", serde(default))]
    pub region: Option<Box<str>>,
}

/// The receipt of a token call, `None` when the core leaves it to the host. `per_call` lowers a row the caller passed
/// for this call alone, for the provider the call is billed as
pub fn priced<T: PricingTable + ?Sized>(
    table: &T,
    call: TokenCall,
    per_call: Option<impl FnOnce(&str) -> T::Row>,
) -> Option<Receipt> {
    let (candidates, named) = named_candidates(table, &call.named, per_call.is_some())?;
    let rows = pricing_rows(table, named, []);
    let (rows, source) = match (per_call, candidates.first()) {
        (Some(lower), Some(first)) => {
            let provider = first.provider.to_string();
            let (rows, at) = rows.with_per_call(PER_CALL_ROW, lower(&provider));
            (
                rows,
                PriceSource::Row {
                    row: at,
                    breakdown_by_name: false,
                },
            )
        }
        _ => (rows, PriceSource::Catalog),
    };
    let query = PriceQuery::Tokens(Box::new(TokensQuery {
        header: Header {
            candidates,
            source,
            basis: Basis {
                service_tier: service_tier_named(call.service_tier.as_deref()),
                data_residency: data_residency(call.data_residency.as_deref()),
                vertex_location: call.vertex_location,
                at: call.at,
            },
            adjustments: call.adjust.map_or(AdjustmentPolicy::None, |key| {
                AdjustmentPolicy::Apply {
                    provider: key.provider,
                    model: key.model,
                    region: key.region,
                }
            }),
            reported: None,
        },
        usage: call.usage,
        requested_model: call.requested_model,
    }));
    match crate::price(&rows, &query) {
        PriceOutcome::Priced(receipt) => Some(*receipt),
        PriceOutcome::Unpriced(_) => None,
    }
}

/// The candidates in main's order, and each one the catalog resolved beside its row
type CallCandidates = (Vec<Candidate>, Vec<(Candidate, RowRef)>);

/// The candidates in main's order and the rows the catalog resolved them to. `None` when main prices the call by a
/// path the catalog does not resolve, no row comes before it and the caller passed no row of its own
fn named_candidates<T: CandidateTable + ?Sized>(
    table: &T,
    named: &Named,
    per_call_row: bool,
) -> Option<CallCandidates> {
    match named {
        Named::Call(call) => {
            let provider: Provider = call.provider.parse().ok()?;
            let found = candidates(table, call);
            let named: Vec<(Candidate, RowRef)> = found
                .rows
                .into_iter()
                .map(|Tried { model, row }| {
                    let candidate = Candidate {
                        model: model.into(),
                        provider: provider.clone(),
                    };
                    (candidate, row)
                })
                .collect();
            (!named.is_empty()).then(|| {
                let candidates = named
                    .iter()
                    .map(|(candidate, _)| candidate.clone())
                    .collect();
                (candidates, named)
            })
        }
        Named::Model {
            model,
            provider,
            region,
        } => {
            let row = match priced_row(table, model, provider, region.as_deref()) {
                Ok(row) => row,
                Err(_) if per_call_row => None,
                Err(_) => return None,
            };
            let candidate = Candidate {
                model: model.as_str().into(),
                provider: Provider::from_str(provider).ok()?,
            };
            Some((
                vec![candidate.clone()],
                row.map(|row| (candidate, row)).into_iter().collect(),
            ))
        }
    }
}

/// The service tier main priced a call at, from the tier name the call carried
pub fn service_tier_named(name: Option<&str>) -> ServiceTier {
    match name.map(str::to_lowercase).as_deref() {
        Some("flex") => ServiceTier::Flex,
        Some("balanced") => ServiceTier::Balanced,
        Some("priority" | "fast") => ServiceTier::Priority,
        Some("ultrafast") => ServiceTier::Ultrafast,
        _ => ServiceTier::Standard,
    }
}

/// `data_residency` lowercased, as the uplift helpers compare it
fn data_residency(written: Option<&str>) -> Option<DataResidency> {
    match written?.to_lowercase().as_str() {
        "eu" => Some(DataResidency::Eu),
        "us" => Some(DataResidency::Us),
        "fedramp" => Some(DataResidency::Fedramp),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    #[rstest]
    #[case::none(None, ServiceTier::Standard)]
    #[case::flex(Some("flex"), ServiceTier::Flex)]
    #[case::upper(Some("PRIORITY"), ServiceTier::Priority)]
    #[case::fast_is_priority(Some("fast"), ServiceTier::Priority)]
    #[case::ultrafast(Some("ultrafast"), ServiceTier::Ultrafast)]
    #[case::balanced(Some("balanced"), ServiceTier::Balanced)]
    #[case::auto_is_standard(Some("auto"), ServiceTier::Standard)]
    #[case::unknown_is_standard(Some("scale"), ServiceTier::Standard)]
    fn a_service_tier_name_picks_the_tier_main_priced_it_at(
        #[case] name: Option<&str>,
        #[case] expected: ServiceTier,
    ) {
        assert_eq!(service_tier_named(name), expected);
    }

    #[rstest]
    #[case::eu(Some("EU"), Some(DataResidency::Eu))]
    #[case::us(Some("us"), Some(DataResidency::Us))]
    #[case::fedramp(Some("FedRAMP"), Some(DataResidency::Fedramp))]
    #[case::unknown(Some("apac"), None)]
    #[case::none(None, None)]
    fn data_residency_is_read_case_blind(
        #[case] written: Option<&str>,
        #[case] expected: Option<DataResidency>,
    ) {
        assert_eq!(data_residency(written), expected);
    }
}
