use serde_json::Value;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum A2ACostError {
    InvalidCost,
}

fn amount(value: &Value) -> Result<f64, A2ACostError> {
    match value {
        Value::Number(number) => number.as_f64().ok_or(A2ACostError::InvalidCost),
        Value::String(number) => number.parse().map_err(|_| A2ACostError::InvalidCost),
        _ => Err(A2ACostError::InvalidCost),
    }
}

pub fn calculate_token_based_cost(
    details: &Value,
    input_rate: Option<&Value>,
    output_rate: Option<&Value>,
) -> Result<f64, A2ACostError> {
    let Some(usage) = details.get("usage") else {
        return Ok(0.0);
    };
    let input_tokens = usage
        .get("prompt_tokens")
        .and_then(Value::as_f64)
        .unwrap_or(0.0);
    let output_tokens = usage
        .get("completion_tokens")
        .and_then(Value::as_f64)
        .unwrap_or(0.0);
    let input = input_rate
        .filter(|rate| !rate.is_null())
        .map(amount)
        .transpose()?
        .unwrap_or(0.0);
    let output = output_rate
        .filter(|rate| !rate.is_null())
        .map(amount)
        .transpose()?
        .unwrap_or(0.0);
    Ok(input_tokens * input + output_tokens * output)
}

pub fn calculate_a2a_cost(details: Option<&Value>) -> Result<f64, A2ACostError> {
    let Some(details) = details else {
        return Ok(0.0);
    };
    if let Some(cost) = details.get("response_cost").filter(|cost| !cost.is_null()) {
        return amount(cost);
    }
    let params = details.get("litellm_params");
    if let Some(cost) = params
        .and_then(|params| params.get("cost_per_query"))
        .filter(|cost| !cost.is_null())
    {
        return amount(cost);
    }
    calculate_token_based_cost(
        details,
        params.and_then(|params| params.get("input_cost_per_token")),
        params.and_then(|params| params.get("output_cost_per_token")),
    )
}
