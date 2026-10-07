use serde_json::{Map, Value};

use crate::{EffectiveRates, Usage, components, schedule, token_counts};

pub struct EstimateRequest<'a> {
    pub usage: Usage,
    pub service_tier: Option<&'a str>,
    pub reasoning_tokens: Option<u64>,
    pub billed_at_ns: Option<i64>,
    pub threshold_is_inclusive: Option<bool>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CatalogCost {
    pub input: f64,
    pub output: f64,
}

#[derive(Clone, Copy)]
enum CatalogRate {
    Missing,
    Null,
    Value(f64),
    Invalid,
}

impl CatalogRate {
    fn read(fields: &Map<String, Value>, key: &str) -> Self {
        match fields.get(key) {
            None => Self::Missing,
            Some(Value::Null) => Self::Null,
            Some(Value::Number(number)) => number.as_f64().map_or(Self::Invalid, Self::Value),
            Some(Value::String(number)) => number.trim().parse().map_or(Self::Invalid, Self::Value),
            _ => Self::Invalid,
        }
    }

    fn or(self, fallback: Self) -> Self {
        match self {
            Self::Missing | Self::Null => fallback,
            _ => self,
        }
    }

    fn checked(self) -> Option<f64> {
        match self {
            Self::Value(value) if value.is_finite() && value >= 0.0 => Some(value),
            _ => None,
        }
    }

    fn used(self, count: u64) -> Option<f64> {
        if count == 0 {
            Some(0.0)
        } else {
            self.checked()
        }
    }
}

const INPUT: &str = "input_cost_per_token";
const OUTPUT: &str = "output_cost_per_token";
const READ: &str = "cache_read_input_token_cost";
const WRITE: &str = "cache_creation_input_token_cost";
const WRITE_1H: &str = "cache_creation_input_token_cost_above_1hr";
const REASONING: &str = "output_cost_per_reasoning_token";

fn service_suffix(tier: Option<&str>) -> &'static str {
    match tier.unwrap_or_default().to_ascii_lowercase().as_str() {
        "flex" => "_flex",
        "balanced" => "_balanced",
        "priority" | "fast" => "_priority",
        "ultrafast" => "_ultrafast",
        _ => "",
    }
}

fn rate(fields: &Map<String, Value>, key: &str, suffix: &str) -> CatalogRate {
    if suffix.is_empty() {
        CatalogRate::read(fields, key)
    } else {
        CatalogRate::read(fields, &format!("{key}{suffix}")).or(CatalogRate::read(fields, key))
    }
}

fn priced_tier(fields: &Map<String, Value>, tokens: u64) -> Option<&Map<String, Value>> {
    if tokens == 0 {
        return None;
    }
    let mut tiers: Vec<_> = fields
        .get("tiered_pricing")?
        .as_array()?
        .iter()
        .filter_map(Value::as_object)
        .filter_map(|tier| {
            let range = tier.get("range")?.as_array()?;
            if range.len() != 2 {
                return None;
            }
            Some((range[0].as_f64()?, range[1].as_f64()?, tier))
        })
        .collect();
    tiers.sort_by(|a, b| a.0.total_cmp(&b.0));
    let selected = tiers
        .iter()
        .find(|(start, end, _)| *start < tokens as f64 && tokens as f64 <= *end)
        .or_else(|| tiers.last())?
        .2;
    selected.contains_key(INPUT).then_some(selected)
}

fn crossed_threshold<'a>(
    fields: &'a Map<String, Value>,
    tokens: u64,
    suffix: &str,
    inclusive: Option<bool>,
) -> Option<&'a str> {
    let inclusive = inclusive
        .unwrap_or_else(|| fields.get("litellm_provider").and_then(Value::as_str) == Some("xai"));
    fields
        .iter()
        .filter_map(|(key, value)| {
            if value.is_null() {
                return None;
            }
            let tail = key.strip_prefix("input_cost_per_token_above_")?;
            let (number, tail) = tail.split_once("_tokens")?;
            if !tail.is_empty() && tail != suffix {
                return None;
            }
            let threshold = if let Some(number) = number.strip_suffix('k') {
                number.parse::<f64>().ok()? * 1000.0
            } else {
                number.parse::<f64>().ok()?
            };
            (tokens as f64 > threshold || (inclusive && tokens as f64 == threshold))
                .then_some((threshold, number))
        })
        .max_by(|a, b| a.0.total_cmp(&b.0))
        .map(|(_, number)| number)
}

fn threshold_rate(
    fields: &Map<String, Value>,
    key: &str,
    suffix: &str,
    threshold: Option<&str>,
) -> CatalogRate {
    let flat = rate(fields, key, suffix);
    threshold.map_or(flat, |threshold| {
        rate(fields, &format!("{key}_above_{threshold}_tokens"), suffix).or(flat)
    })
}

fn off_peak<'a>(
    fields: &'a Map<String, Value>,
    request: &EstimateRequest<'_>,
) -> Option<Option<&'a Map<String, Value>>> {
    let Some(value) = fields.get("off_peak_pricing") else {
        return Some(None);
    };
    if value.is_null() || value.as_object().is_some_and(Map::is_empty) {
        return Some(None);
    }
    let start = request.billed_at_ns?;
    Some(
        value
            .as_object()
            .filter(|block| schedule::active(block, start)),
    )
}

fn overlay(block: Option<&Map<String, Value>>, key: &str, base: CatalogRate) -> CatalogRate {
    block.map_or(base, |block| CatalogRate::read(block, key).or(base))
}

pub fn estimate(fields: &Map<String, Value>, request: &EstimateRequest<'_>) -> Option<f64> {
    calculate(fields, request).map(|cost| cost.input + cost.output)
}

pub fn calculate(
    fields: &Map<String, Value>,
    request: &EstimateRequest<'_>,
) -> Option<CatalogCost> {
    let usage = request.usage;
    let (regular, tokens, writes) = token_counts(usage).ok()?;
    let suffix = service_suffix(request.service_tier);
    let off_peak = off_peak(fields, request)?;
    let tier = priced_tier(fields, tokens);
    let threshold = crossed_threshold(fields, tokens, suffix, request.threshold_is_inclusive);
    let select = |key| threshold_rate(fields, key, suffix, threshold);
    let fireworks = fields.get("litellm_provider").and_then(Value::as_str) == Some("fireworks_ai");
    let default_read = if fireworks
        && matches!(
            CatalogRate::read(fields, READ),
            CatalogRate::Missing | CatalogRate::Null
        ) {
        CatalogRate::read(fields, INPUT)
            .checked()
            .map(|value| CatalogRate::Value(value * 0.5))
    } else {
        None
    };
    let (input, output, read, write, write_1h) = if let Some(tier) = tier {
        let input = CatalogRate::read(tier, INPUT);
        let output = if tier.contains_key(OUTPUT) {
            CatalogRate::read(tier, OUTPUT)
        } else {
            CatalogRate::read(fields, OUTPUT)
        };
        let write = CatalogRate::read(tier, WRITE).or(input);
        (
            input,
            output,
            CatalogRate::read(tier, READ).or(input),
            write,
            CatalogRate::read(tier, WRITE_1H).or(write),
        )
    } else {
        let input = select(INPUT);
        let output = select(OUTPUT);
        let output = if matches!(output, CatalogRate::Value(0.0)) {
            CatalogRate::read(fields, "output_cost_per_image_token").or(output)
        } else {
            output
        };
        let fallback = overlay(off_peak, INPUT, input);
        let read = select(READ).or(default_read.unwrap_or(fallback));
        let write = select(WRITE).or(fallback);
        (
            input,
            output,
            read,
            write,
            threshold.map_or(CatalogRate::read(fields, WRITE_1H), |threshold| {
                rate(
                    fields,
                    &format!("{WRITE_1H}_above_{threshold}_tokens"),
                    suffix,
                )
                .or(CatalogRate::read(fields, WRITE_1H))
            }),
        )
    };
    let input = overlay(off_peak, INPUT, input);
    let output = overlay(off_peak, OUTPUT, output);
    let read = if let (Some(block), Some(default)) = (off_peak, default_read) {
        if !block.contains_key(READ) {
            CatalogRate::read(block, INPUT)
                .checked()
                .map_or(default, |value| CatalogRate::Value(value * 0.5))
        } else {
            overlay(off_peak, READ, read)
        }
    } else {
        overlay(off_peak, READ, read)
    };
    let write = overlay(off_peak, WRITE, write);
    let write_1h = write_1h.or(write);
    // Without a duration split, either write price may apply to the reported total.
    if usage.cache_write_tokens > 0
        && usage.cache_write_5m_tokens.is_none()
        && write.checked()? != write_1h.checked()?
    {
        return None;
    }
    let rates = EffectiveRates {
        input: input.checked()?,
        output: output.checked()?,
        cache_read: read.used(usage.cache_read_tokens)?,
        cache_write_5m: write.used(writes.0)?,
        cache_write_1h: write_1h.used(writes.1)?,
    };
    let reasoning_tokens = request.reasoning_tokens.unwrap_or_default();
    if reasoning_tokens > usage.completion_tokens {
        return None;
    }
    let reasoning = if reasoning_tokens > 0
        || (request.reasoning_tokens.is_none() && usage.completion_tokens > 0)
    {
        let standard = CatalogRate::read(fields, REASONING);
        let service = CatalogRate::read(fields, &format!("{REASONING}{suffix}"));
        let service_output = !suffix.is_empty()
            && !matches!(
                CatalogRate::read(fields, &format!("{OUTPUT}{suffix}")),
                CatalogRate::Missing | CatalogRate::Null
            );
        let base = service.or(if service_output {
            output
        } else {
            standard.or(output)
        });
        let tier_reasoning = tier.map_or(base, |tier| {
            CatalogRate::read(tier, REASONING)
                .or(CatalogRate::read(tier, OUTPUT))
                .or(base)
        });
        overlay(off_peak, REASONING, tier_reasoning).checked()?
    } else {
        rates.output
    };
    if request.reasoning_tokens.is_none()
        && usage.completion_tokens > 0
        && reasoning != rates.output
    {
        return None;
    }
    components(
        usage,
        regular,
        writes,
        rates,
        1.0,
        reasoning_tokens,
        reasoning,
    )
    .ok()
    .map(|cost| CatalogCost {
        input: cost.input(),
        output: cost.output(),
    })
}
