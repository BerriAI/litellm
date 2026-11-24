//! A Rust-priced receipt as Python's logging reads it: the totals and the token-type breakdown main stores on the
//! logging object, in main's field names. Both the pricing bridge and the route hand-off give Python this one object

use litellm_cost::receipt::{BilledRates, Receipt};
use pyo3::prelude::*;

/// The per-meter amounts and rates of a priced call, in the fields main stores in its cost breakdown
#[pyclass(
    frozen,
    get_all,
    name = "NativeReceipt",
    module = "litellm.rust_bridge._native"
)]
pub struct NativeReceipt {
    /// The candidate model that priced
    model: Option<String>,
    prompt_cost: f64,
    completion_cost: f64,
    tools_cost: f64,
    original_cost: f64,
    discount_percent: f64,
    discount_amount: f64,
    margin_percent: f64,
    margin_fixed_amount: f64,
    margin_total_amount: f64,
    cache_read_cost: f64,
    cache_creation_cost: f64,
    reasoning_cost: f64,
    rates: Option<Py<NativeBilledRates>>,
    cache_savings: Option<f64>,
    total: f64,
}

/// `BilledTokenRates`: the per-token rates the breakdown was billed at
#[pyclass(
    frozen,
    get_all,
    name = "NativeBilledRates",
    module = "litellm.rust_bridge._native"
)]
pub struct NativeBilledRates {
    input: f64,
    output: f64,
    cache_read: f64,
    cache_read_audio: f64,
    cache_read_image: Option<f64>,
    cache_write: f64,
    cache_write_1h: f64,
    cache_write_12h: f64,
    reasoning: f64,
}

impl NativeReceipt {
    pub fn new(py: Python<'_>, receipt: &Receipt) -> PyResult<Self> {
        let rates = receipt
            .rates
            .map(|rates| Py::new(py, NativeBilledRates::from(rates)))
            .transpose()?;
        Ok(Self {
            original_cost: receipt.original(),
            model: receipt
                .candidate
                .as_ref()
                .map(|candidate| candidate.model.to_string()),
            prompt_cost: receipt.prompt.get(),
            completion_cost: receipt.completion.get(),
            tools_cost: receipt.tools.get(),
            discount_percent: receipt.discount.percent,
            discount_amount: receipt.discount.amount,
            margin_percent: receipt.margin.percent,
            margin_fixed_amount: receipt.margin.fixed,
            margin_total_amount: receipt.margin.total,
            cache_read_cost: receipt.breakdown.cache_read,
            cache_creation_cost: receipt.breakdown.cache_creation,
            reasoning_cost: receipt.breakdown.reasoning,
            rates,
            cache_savings: receipt.cache_savings,
            total: receipt.total,
        })
    }
}

impl From<BilledRates> for NativeBilledRates {
    fn from(rates: BilledRates) -> Self {
        Self {
            input: rates.input,
            output: rates.output,
            cache_read: rates.cache_read,
            cache_read_audio: rates.cache_read_audio,
            cache_read_image: rates.cache_read_image,
            cache_write: rates.cache_write,
            cache_write_1h: rates.cache_write_one_hour,
            cache_write_12h: rates.cache_write_twelve_hours,
            reasoning: rates.reasoning,
        }
    }
}
