//! The customer surcharge on the ledger path, the Rust port of
//! `litellm/proxy/spend_tracking/customer_cost_margin.py`: `apply_callback_customer_cost_margin`
//! on a success, and `_failure_row_with_customer_surcharge` on each failure row.
//!
//! The margin itself (`request_customer_cost_margin`, which reads the gateway's
//! `cost_margin_config` and the request's customer and app) arrives resolved on each row
//! ([`RowRequest::customer_margin`]); the surcharge, the breakdown it records and the cost it
//! charges are computed here. The surcharged cost is what the row's `spend`, the ledger columns,
//! the accounting payload and the counters all carry, and the breakdown is written where the row
//! builder reads it: the logging payload's `cost_breakdown`, or the row's
//! `snapshotless_cost_breakdown` when the row has no payload.

use serde_json::{Map, Value as Json, json};

use crate::{hook_input::RowRequest, search_free_tier::validated_breakdown};

const SNAPSHOTLESS_COST_BREAKDOWN: &str = "snapshotless_cost_breakdown";

/// `apply_customer_cost_margin` over `finalize_cost(cost, breakdown)`: the surcharged cost and
/// its breakdown, or `None` when nothing is added (no margin, a zero margin, or a breakdown that
/// already records a surcharge).
fn surcharged(
    breakdown: Option<&Map<String, Json>>,
    cost: f64,
    margin: Option<f64>,
) -> Option<(f64, Map<String, Json>)> {
    let margin = margin.filter(|margin| *margin != 0.0)?;
    if breakdown.is_some_and(|breakdown| breakdown.contains_key("customer_margin_amount")) {
        return None;
    }
    let surcharge = cost * margin;
    let amount = cost + surcharge;
    let mut recorded = breakdown.cloned().unwrap_or_default();
    recorded.insert("customer_margin_percent".to_owned(), json!(margin));
    recorded.insert("customer_margin_amount".to_owned(), json!(surcharge));
    recorded.insert("total_cost".to_owned(), json!(amount));
    Some((amount, recorded))
}

/// `apply_callback_customer_cost_margin` on the row's logging payload: a payload whose cost is
/// already finalized, or a row with no payload, keeps its cost. Otherwise the payload's
/// `response_cost` and `cost_breakdown` take the surcharge, and the surcharged cost is returned.
fn surcharge_payload(row: &mut RowRequest, cost: f64) -> f64 {
    if row.cost_finalized {
        return cost;
    }
    let margin = row.customer_margin;
    let Some(payload) = row
        .row
        .get_mut("standard_logging_object")
        .and_then(Json::as_object_mut)
    else {
        return cost;
    };
    let Some((amount, breakdown)) = surcharged(
        payload.get("cost_breakdown").and_then(Json::as_object),
        cost,
        margin,
    ) else {
        return cost;
    };
    payload.insert("response_cost".to_owned(), json!(amount));
    payload.insert("cost_breakdown".to_owned(), validated_breakdown(&breakdown));
    amount
}

/// The success's charged cost once the customer surcharge is applied, after the search free
/// tier, as the cost callback applies it.
pub(crate) fn on_success(row: &mut RowRequest, cost: f64) -> f64 {
    surcharge_payload(row, cost)
}

/// `_failure_row_with_customer_surcharge`: a failure row the provider billed is charged to the
/// customer like a completed request. A row with no logging payload records its surcharge as
/// the row's `snapshotless_cost_breakdown`.
pub(crate) fn on_failure_row(row: &mut RowRequest, cost: f64) -> f64 {
    let Some(margin) = row.customer_margin else {
        return cost;
    };
    if cost <= 0.0 {
        return cost;
    }
    if crate::policy::payload(&row.row).is_some() {
        return surcharge_payload(row, cost);
    }
    let (amount, breakdown) = surcharged(None, cost, Some(margin))
        .map_or((cost, Json::Null), |(amount, breakdown)| {
            (amount, validated_breakdown(&breakdown))
        });
    if let Some(input) = row.row.as_object_mut() {
        input.insert(SNAPSHOTLESS_COST_BREAKDOWN.to_owned(), breakdown);
    }
    amount
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    fn row(payload: Json, margin: Option<f64>, cost_finalized: bool) -> RowRequest {
        RowRequest {
            row: json!({"kwargs": {"response_cost": 2.0}, "standard_logging_object": payload}),
            customer_margin: margin,
            cost_finalized,
            ..RowRequest::default()
        }
    }

    #[rstest]
    #[case::no_margin(None, 2.0)]
    #[case::zero_margin(Some(0.0), 2.0)]
    #[case::ten_percent(Some(0.1), 2.2)]
    #[case::integer_margin(Some(1.0), 4.0)]
    fn a_success_is_surcharged_by_its_margin(#[case] margin: Option<f64>, #[case] charged: f64) {
        let mut request = row(
            json!({"cost_breakdown": {"input_cost": 1, "output_cost": 1.0}}),
            margin,
            false,
        );

        let cost = on_success(&mut request, 2.0);

        assert!((cost - charged).abs() < 1e-12, "{cost} != {charged}");
        let payload = &request.row["standard_logging_object"];
        if charged == 2.0 {
            assert_eq!(
                payload,
                &json!({"cost_breakdown": {"input_cost": 1, "output_cost": 1.0}})
            );
            return;
        }
        let margin = margin.unwrap();
        assert_eq!(
            payload,
            &json!({
                "cost_breakdown": {
                    "customer_margin_percent": margin,
                    "customer_margin_amount": 2.0 * margin,
                    "input_cost": 1.0,
                    "output_cost": 1.0,
                    "total_cost": 2.0 + 2.0 * margin,
                },
                "response_cost": 2.0 + 2.0 * margin,
            })
        );
        assert_eq!(request.row["kwargs"]["response_cost"], json!(2.0));
    }

    #[test]
    fn an_integer_cost_breakdown_records_the_surcharge_as_floats() {
        let mut request = row(
            json!({"cost_breakdown": {"total_cost": 3}}),
            Some(0.5),
            false,
        );

        assert_eq!(on_success(&mut request, 3.0), 4.5);

        assert_eq!(
            request.row["standard_logging_object"]["cost_breakdown"],
            json!({"customer_margin_percent": 0.5, "customer_margin_amount": 1.5, "total_cost": 4.5})
        );
    }

    #[rstest]
    #[case::finalized(json!({"cost_breakdown": null}), true)]
    #[case::already_surcharged(json!({"cost_breakdown": {"customer_margin_amount": 1, "total_cost": 2.0}}), false)]
    fn a_settled_cost_is_not_surcharged_again(#[case] payload: Json, #[case] cost_finalized: bool) {
        let mut request = row(payload.clone(), Some(0.1), cost_finalized);

        assert_eq!(on_success(&mut request, 2.0), 2.0);

        assert_eq!(request.row["standard_logging_object"], payload);
    }

    #[test]
    fn a_success_with_no_payload_keeps_its_cost() {
        let mut request = row(Json::Null, Some(0.1), false);

        assert_eq!(on_success(&mut request, 2.0), 2.0);

        assert!(request.row.get(SNAPSHOTLESS_COST_BREAKDOWN).is_none());
    }

    #[rstest]
    #[case::ten_percent(Some(0.1), 2.0, 2.2)]
    #[case::integer_margin(Some(2.0), 1.0, 3.0)]
    #[case::no_margin(None, 2.0, 2.0)]
    #[case::unbilled(Some(0.1), 0.0, 0.0)]
    fn a_failure_row_with_a_payload_is_surcharged_like_a_success(
        #[case] margin: Option<f64>,
        #[case] cost: f64,
        #[case] charged: f64,
    ) {
        let mut request = row(json!({"cost_breakdown": null}), margin, false);

        let surcharged = on_failure_row(&mut request, cost);

        assert!(
            (surcharged - charged).abs() < 1e-12,
            "{surcharged} != {charged}"
        );
        let recorded = &request.row["standard_logging_object"]["cost_breakdown"];
        if charged == cost {
            assert!(recorded.is_null());
        } else {
            let margin = margin.unwrap();
            assert_eq!(recorded["customer_margin_amount"], json!(cost * margin));
            assert_eq!(recorded["total_cost"], json!(cost + cost * margin));
        }
        assert!(request.row.get(SNAPSHOTLESS_COST_BREAKDOWN).is_none());
    }

    #[rstest]
    #[case::ten_percent(Some(0.1), 2.0, 2.2, json!({"customer_margin_percent": 0.1, "customer_margin_amount": 0.2, "total_cost": 2.2}))]
    #[case::integer_margin(Some(1.0), 3.0, 6.0, json!({"customer_margin_percent": 1.0, "customer_margin_amount": 3.0, "total_cost": 6.0}))]
    #[case::zero_margin(Some(0.0), 2.0, 2.0, Json::Null)]
    fn a_failure_row_with_no_payload_records_a_snapshotless_breakdown(
        #[case] margin: Option<f64>,
        #[case] cost: f64,
        #[case] charged: f64,
        #[case] breakdown: Json,
    ) {
        let mut request = row(Json::Null, margin, false);

        let surcharged = on_failure_row(&mut request, cost);

        assert!(
            (surcharged - charged).abs() < 1e-12,
            "{surcharged} != {charged}"
        );
        let recorded = &request.row[SNAPSHOTLESS_COST_BREAKDOWN];
        if breakdown.is_null() {
            assert!(recorded.is_null());
        } else {
            for key in [
                "customer_margin_percent",
                "customer_margin_amount",
                "total_cost",
            ] {
                let (got, want) = (
                    recorded[key].as_f64().unwrap(),
                    breakdown[key].as_f64().unwrap(),
                );
                assert!((got - want).abs() < 1e-12, "{key}: {got} != {want}");
            }
        }
    }

    #[test]
    fn a_failure_row_with_no_margin_keeps_its_snapshotless_breakdown() {
        let mut request = row(Json::Null, None, false);

        assert_eq!(on_failure_row(&mut request, 2.0), 2.0);

        assert!(request.row.get(SNAPSHOTLESS_COST_BREAKDOWN).is_none());
    }
}
