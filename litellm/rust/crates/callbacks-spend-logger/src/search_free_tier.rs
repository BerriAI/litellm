//! The search free tier (`litellm/proxy/hooks/search_free_tier.py`): the grounding queries that
//! land under a model's shared free allowance (`search_free_requests` in its model info) are not
//! charged. The allowance is counted in the internal usage cache's Redis, the counter the Python
//! path shares, through the `increment_search_free_tier` host call. Everything else (the units,
//! the counter key and TTL, the freed cost and the cost breakdown) is decided here from the row
//! builder input.

use chrono::{DateTime, Datelike, Duration, NaiveDate, Utc};
use serde_json::{Map, Value as Json, json};

const TTL_GRACE_SECONDS: i64 = 3600;
/// `CostBreakdown`'s keys in declaration order, which its validation writes them in, keeping
/// none other.
const BREAKDOWN_KEYS: [&str; 20] = [
    "customer_margin_percent",
    "customer_margin_amount",
    "service_tier",
    "data_residency",
    "vertex_location",
    "input_cost",
    "cache_read_cost",
    "cache_creation_cost",
    "output_cost",
    "reasoning_cost",
    "total_cost",
    "tool_usage_cost",
    "guardrail_cost",
    "additional_costs",
    "original_cost",
    "discount_percent",
    "discount_amount",
    "margin_percent",
    "margin_fixed_amount",
    "margin_total_amount",
];
const BREAKDOWN_TEXT_KEYS: [&str; 3] = ["service_tier", "data_residency", "vertex_location"];
const DEFAULT_COST_PER_QUERY: f64 = 0.035;

/// The allowance one request draws from: the shared counter, how many units the request adds
/// and how much the whole grounding costs.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Draw {
    pub key: String,
    pub units: i64,
    pub ttl: i64,
    pub allowance: i64,
    pub grounding_cost: f64,
}

#[derive(Clone, Copy)]
enum Period {
    Day,
    Month,
}

fn count(value: Option<&Json>) -> i64 {
    match value {
        Some(Json::Number(number)) => number
            .as_i64()
            .or_else(|| number.as_f64().map(|number| number as i64))
            .unwrap_or(0),
        _ => 0,
    }
}

/// `_grounding_queries`: the prompt details' `web_search_requests`, else `server_tool_use`'s.
fn grounding_queries(usage: &Json) -> i64 {
    let from_details = count(
        usage
            .get("prompt_tokens_details")
            .and_then(|details| details.get("web_search_requests")),
    );
    if from_details != 0 {
        return from_details;
    }
    count(
        usage
            .get("server_tool_use")
            .and_then(|tool_use| tool_use.get("web_search_requests")),
    )
}

/// `cost_per_web_search_request`: per grounded prompt unless the model bills per query.
fn grounding_cost(queries: i64, model_info: &Json) -> f64 {
    let per_query = model_info
        .get("search_context_cost_per_query")
        .and_then(|costs| costs.get("search_context_size_medium"))
        .and_then(Json::as_f64)
        .unwrap_or(DEFAULT_COST_PER_QUERY);
    let billing = model_info
        .get("web_search_billing_unit")
        .and_then(Json::as_str)
        .filter(|unit| !unit.is_empty())
        .unwrap_or("per_prompt");
    let billable = if queries > 0 && billing == "per_prompt" {
        1
    } else {
        queries
    };
    per_query * billable as f64
}

/// The UTC instant the period after `now` starts: tomorrow's midnight, or the 1st of next month.
fn next_reset(period: Period, now: DateTime<Utc>) -> Option<DateTime<Utc>> {
    let today = now.date_naive();
    let start = match period {
        Period::Day => today.succ_opt()?,
        Period::Month if today.month() == 12 => NaiveDate::from_ymd_opt(today.year() + 1, 1, 1)?,
        Period::Month => NaiveDate::from_ymd_opt(today.year(), today.month() + 1, 1)?,
    };
    Some(start.and_hms_opt(0, 0, 0)?.and_utc())
}

/// The allowance this request draws from, or `None` when it has none to draw: a model with no
/// free allowance, a cache hit, or no grounding queries.
pub(crate) fn draw(row: &Json, cache_hit: bool, now: DateTime<Utc>) -> Option<Draw> {
    let payload = row.get("standard_logging_object")?;
    let model_info = payload
        .get("model_map_information")?
        .get("model_map_value")
        .filter(|info| !info.is_null())?;
    let allowance = model_info
        .get("search_free_requests")
        .filter(|allowance| !allowance.is_null())?;
    let usage = row.get("usage").filter(|usage| usage.is_object())?;
    if cache_hit {
        return None;
    }
    let queries = grounding_queries(usage);
    if queries == 0 {
        return None;
    }
    let units = if model_info
        .get("web_search_billing_unit")
        .and_then(Json::as_str)
        == Some("per_query")
    {
        queries
    } else {
        1
    };
    let (period, stamp) = match allowance.get("period").and_then(Json::as_str) {
        Some("day") => (Period::Day, now.format("%Y-%m-%d")),
        Some("month") => (Period::Month, now.format("%Y-%m")),
        other => {
            tracing::warn!(period = ?other, "search free tier period is neither day nor month, grounding is billed at full price");
            return None;
        }
    };
    let provider = payload
        .get("custom_llm_provider")
        .and_then(Json::as_str)
        .filter(|provider| !provider.is_empty())
        .or_else(|| model_info.get("litellm_provider").and_then(Json::as_str))?;
    let group = allowance.get("shared_group").and_then(Json::as_str)?;
    let ttl = (next_reset(period, now)? - now)
        .max(Duration::zero())
        .num_seconds()
        + TTL_GRACE_SECONDS;
    Some(Draw {
        key: format!("litellm:search_free_tier:{provider}:{group}:{stamp}"),
        units,
        ttl,
        allowance: count(allowance.get("count")),
        grounding_cost: grounding_cost(queries, model_info),
    })
}

/// `freed_units`: the units of this request that landed under the allowance, given the shared
/// total after they were added.
fn freed_units(allowance: i64, units: i64, total_after: i64) -> i64 {
    (allowance - (total_after - units)).min(units).max(0)
}

/// The discount the allowance gives this request, capped at its cost.
pub(crate) fn discount(draw: &Draw, cost: f64, total_after: f64) -> f64 {
    let freed = freed_units(draw.allowance, draw.units, total_after as i64);
    cost.min(draw.grounding_cost * freed as f64 / draw.units as f64)
}

fn number(value: Option<&Json>) -> Option<f64> {
    value.and_then(Json::as_f64)
}

fn as_float(value: &Json) -> Json {
    match value {
        Json::Number(number) => number
            .as_f64()
            .map_or_else(|| value.clone(), |number| json!(number)),
        Json::Object(entries) => Json::Object(
            entries
                .iter()
                .map(|(key, value)| (key.clone(), as_float(value)))
                .collect(),
        ),
        other => other.clone(),
    }
}

/// `TypeAdapter(CostBreakdown).validate_python`: the declared keys in declaration order, the
/// costs as floats.
pub(crate) fn validated_breakdown(breakdown: &Map<String, Json>) -> Json {
    Json::Object(
        BREAKDOWN_KEYS
            .iter()
            .filter_map(|key| {
                let value = breakdown.get(*key)?;
                let value = if BREAKDOWN_TEXT_KEYS.contains(key) {
                    value.clone()
                } else {
                    as_float(value)
                };
                Some(((*key).to_owned(), value))
            })
            .collect(),
    )
}

/// `finalize_cost` written back the way `record_freed_cost` writes it: the request's cost, the
/// payload's `response_cost` and, for a real discount, the breakdown's total, original cost and
/// accumulated discount.
pub(crate) fn apply(row: &mut Json, cost: f64, allowance_discount: f64) -> f64 {
    let discount = cost.max(0.0).min(allowance_discount.max(0.0));
    let total = cost - discount;
    if let Some(kwargs) = row.get_mut("kwargs").and_then(Json::as_object_mut) {
        kwargs.insert("response_cost".to_owned(), json!(total));
    }
    let Some(payload) = row
        .get_mut("standard_logging_object")
        .and_then(Json::as_object_mut)
    else {
        return total;
    };
    payload.insert("response_cost".to_owned(), json!(total));
    let existing = payload
        .get("cost_breakdown")
        .and_then(Json::as_object)
        .cloned();
    let breakdown = match existing {
        Some(mut breakdown) if discount > 0.0 => {
            let original = number(breakdown.get("original_cost")).unwrap_or(cost);
            let accumulated = number(breakdown.get("discount_amount")).unwrap_or(0.0) + discount;
            breakdown.insert("total_cost".to_owned(), json!(total));
            breakdown.insert("original_cost".to_owned(), json!(original));
            breakdown.insert("discount_amount".to_owned(), json!(accumulated));
            breakdown
        }
        Some(breakdown) => breakdown,
        None if discount > 0.0 => Map::from_iter([
            ("total_cost".to_owned(), json!(total)),
            ("original_cost".to_owned(), json!(cost)),
            ("discount_amount".to_owned(), json!(discount)),
        ]),
        None => return total,
    };
    payload.insert("cost_breakdown".to_owned(), validated_breakdown(&breakdown));
    total
}

#[cfg(test)]
mod tests {
    use chrono::TimeZone;
    use rstest::rstest;

    use super::*;

    fn row(unit: &str, queries: i64, period: &str) -> Json {
        json!({
            "usage": {"prompt_tokens_details": {"web_search_requests": queries}},
            "standard_logging_object": {
                "custom_llm_provider": "gemini",
                "cost_breakdown": {"input_cost": 0.5},
                "response_cost": 1.0,
                "model_map_information": {"model_map_value": {
                    "litellm_provider": "vertex_ai",
                    "web_search_billing_unit": unit,
                    "search_context_cost_per_query": {"search_context_size_medium": 0.014},
                    "search_free_requests": {"count": 5, "period": period, "shared_group": "g3"},
                }},
            },
            "kwargs": {"response_cost": 1.0},
        })
    }

    fn at(year: i32, month: u32, day: u32, hour: u32) -> DateTime<Utc> {
        Utc.with_ymd_and_hms(year, month, day, hour, 0, 0).unwrap()
    }

    #[rstest]
    #[case::daily("day", at(2026, 3, 9, 23), "litellm:search_free_tier:gemini:g3:2026-03-09", 3600 + 3600)]
    #[case::monthly_rolls_the_year("month", at(2026, 12, 31, 0), "litellm:search_free_tier:gemini:g3:2026-12", 24 * 3600 + 3600)]
    fn the_counter_key_and_ttl_follow_the_period(
        #[case] period: &str,
        #[case] now: DateTime<Utc>,
        #[case] key: &str,
        #[case] ttl: i64,
    ) {
        let draw = draw(&row("per_query", 3, period), false, now).unwrap();
        assert_eq!((draw.key.as_str(), draw.ttl), (key, ttl));
    }

    #[rstest]
    #[case::per_query("per_query", 3, 0.042)]
    #[case::per_prompt("per_prompt", 1, 0.014)]
    fn units_and_grounding_cost_follow_the_billing_unit(
        #[case] unit: &str,
        #[case] units: i64,
        #[case] cost: f64,
    ) {
        let draw = draw(&row(unit, 3, "day"), false, at(2026, 3, 9, 1)).unwrap();
        assert_eq!(draw.units, units);
        assert!((draw.grounding_cost - cost).abs() < 1e-12);
    }

    #[test]
    fn no_draw_for_a_cache_hit_or_no_grounding() {
        let now = at(2026, 3, 9, 1);
        assert_eq!(draw(&row("per_query", 3, "day"), true, now), None);
        assert_eq!(draw(&row("per_query", 0, "day"), false, now), None);
    }

    #[rstest]
    #[case::all_under(3, 3, 1.0)]
    #[case::straddles(6, 3, 2.0 / 3.0)]
    #[case::all_over(9, 3, 0.0)]
    fn only_the_units_under_the_allowance_are_freed(
        #[case] total_after: i64,
        #[case] units: i64,
        #[case] share: f64,
    ) {
        let draw = Draw {
            key: String::new(),
            units,
            ttl: 0,
            allowance: 5,
            grounding_cost: 0.3,
        };
        assert!((discount(&draw, 1.0, total_after as f64) - 0.3 * share).abs() < 1e-12);
    }

    #[test]
    fn a_discount_rewrites_the_cost_and_accumulates_in_the_breakdown() {
        let mut request = row("per_query", 3, "day");
        let total = apply(&mut request, 1.0, 0.25);
        assert_eq!(total, 0.75);
        assert_eq!(request["kwargs"]["response_cost"], json!(0.75));
        let payload = &request["standard_logging_object"];
        assert_eq!(payload["response_cost"], json!(0.75));
        assert_eq!(
            payload["cost_breakdown"],
            json!({"input_cost": 0.5, "total_cost": 0.75, "original_cost": 1.0, "discount_amount": 0.25})
        );
        let keys: Vec<&String> = payload["cost_breakdown"]
            .as_object()
            .unwrap()
            .keys()
            .collect();
        assert_eq!(
            keys,
            [
                "input_cost",
                "total_cost",
                "original_cost",
                "discount_amount"
            ]
        );
    }

    #[test]
    fn no_discount_leaves_the_breakdown_alone() {
        let mut request = row("per_query", 3, "day");
        assert_eq!(apply(&mut request, 1.0, 0.0), 1.0);
        assert_eq!(
            request["standard_logging_object"]["cost_breakdown"],
            json!({"input_cost": 0.5})
        );
    }
}
