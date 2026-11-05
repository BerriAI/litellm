use litellm_python_compat::Value;

use super::keys::Counter;
use crate::{DecodeError, FromValue, Number, codec, value};

/// `(deployment_id, model, minute, kind)`, the kind spelled as its key segment.
impl FromValue for Counter {
    fn from_value(value: &Value) -> Result<Self, DecodeError> {
        const SHAPE: &str = "a usage counter is (deployment_id, model, minute, \"tpm\" | \"rpm\")";
        let [deployment_id, model, minute, kind] = value::fields(value, SHAPE)?;
        Ok(Self {
            deployment_id: value::text(deployment_id, SHAPE)?,
            model: value::optional_text(model, SHAPE)?,
            minute: value::text(minute, SHAPE)?,
            kind: value::text(kind, SHAPE)?
                .parse()
                .map_err(|_| DecodeError::Shape(SHAPE))?,
        })
    }
}

pub(super) fn decode(raw: &[u8]) -> Result<Number, DecodeError> {
    codec::number(raw, "a usage counter is a number")
}
