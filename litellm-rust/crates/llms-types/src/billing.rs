use crate::InvalidBilledAmount;

use serde_json::{Number, Value};

/// An amount a provider reports having billed for a call, in US dollars. Hosts send it as a
/// JSON number or a numeric string; it is finite and not negative, or it is not an amount.
#[macro_rules_attribute::apply(crate::wire_type)]
#[serde(try_from = "Value", into = "Number")]
pub struct BilledAmount(Number);

impl BilledAmount {
    pub fn as_f64(&self) -> f64 {
        self.0.as_f64().unwrap_or(0.0)
    }

    pub fn into_number(self) -> Number {
        self.0
    }
}

impl TryFrom<Value> for BilledAmount {
    type Error = InvalidBilledAmount;

    fn try_from(value: Value) -> Result<Self, Self::Error> {
        let amount = match &value {
            Value::Number(amount) => amount.as_f64(),
            Value::String(amount) => amount.parse::<f64>().ok(),
            _ => None,
        };
        amount
            .filter(|amount| *amount >= 0.0)
            .and_then(Number::from_f64)
            .map(Self)
            .ok_or(InvalidBilledAmount(value))
    }
}

impl From<BilledAmount> for Number {
    fn from(amount: BilledAmount) -> Self {
        amount.0
    }
}
