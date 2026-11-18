use serde_json::Value;

use crate::snapshot::{LiveTable, RowRef, Snapshot};

/// What an image call asks for. `resolve_image` builds main's eight key spellings from it
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ImageCandidates {
    pub model: String,
    pub provider: Option<String>,
    /// Width and height, `1024 x 1024` when the request names none, as on main
    pub size: Option<(u32, u32)>,
    pub quality: Option<String>,
}

/// fal's size keyed rows, `fal_ai/{quality}/{W}-x-{H}/{model}` (`_keyed_rows` and `_keyed_cost_per_image` in
/// `llms/fal_ai/cost_calculator.py`)
impl<T: LiveTable> Snapshot<T> {
    /// The `quality` row of `model` whose pixel count is nearest `target`'s, the smaller on a tie and the first in row
    /// order after that, among the rows listing a numeric `output_cost_per_image`. `None` when the model has none
    pub fn fal_sized_row(&self, model: &str, quality: &str, target: (u64, u64)) -> Option<RowRef> {
        let prefix = format!("fal_ai/{quality}/");
        let suffix = format!("/{model}");
        let wanted = target.0 * target.1;
        self.table
            .keys_with(&prefix, &suffix)
            .into_iter()
            .filter_map(|key| {
                let size = key.get(prefix.len()..key.len().checked_sub(suffix.len())?)?;
                let pixels = fal_pixels(size)?;
                let priced = self
                    .fields(&key)?
                    .get("output_cost_per_image")
                    .is_some_and(Value::is_number);
                priced.then_some((wanted.abs_diff(pixels), pixels, key))
            })
            .min_by_key(|(distance, pixels, _)| (*distance, *pixels))
            .and_then(|(_, _, key)| self.row_at(&key))
    }
}

/// `_parse_keyed_dimensions`: `{W}-x-{H}` with both sides positive, as a pixel count
fn fal_pixels(size: &str) -> Option<u64> {
    let (width, height) = size.split_once("-x-")?;
    let side = |text: &str| text.parse::<u64>().ok().filter(|side| *side > 0);
    Some(side(width)? * side(height)?)
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use indexmap::IndexMap;
    use rstest::rstest;
    use serde_json::json;

    use crate::test_support::{TestRow, TestRows, names, snapshot};

    fn rows(keys: &[(&str, serde_json::Value)]) -> TestRows {
        TestRows(
            keys.iter()
                .map(|(key, price)| {
                    let fields =
                        json!({"litellm_provider": "fal_ai", "output_cost_per_image": price});
                    let fields = fields.as_object().unwrap().clone().into_iter().collect();
                    ((*key).to_owned(), TestRow::new(fields, BTreeSet::new()))
                })
                .collect::<IndexMap<_, _>>(),
        )
    }

    fn picked(
        keys: &[(&str, serde_json::Value)],
        quality: &str,
        target: (u64, u64),
    ) -> Option<String> {
        snapshot(rows(keys), names(&["fal_ai"]))
            .fal_sized_row("openai/gpt-image-2", quality, target)
            .map(|row| row.key)
    }

    #[rstest]
    #[case::exact((1024, 1024), "fal_ai/high/1024-x-1024/openai/gpt-image-2")]
    #[case::nearest_above((1200, 1200), "fal_ai/high/1024-x-1536/openai/gpt-image-2")]
    #[case::nearest_below((900, 900), "fal_ai/high/1024-x-768/openai/gpt-image-2")]
    #[case::tie_takes_the_smaller((1024, 896), "fal_ai/high/1024-x-768/openai/gpt-image-2")]
    fn the_row_nearest_the_target_pixels_wins(#[case] target: (u64, u64), #[case] expected: &str) {
        let keys = [
            ("fal_ai/high/1024-x-1024/openai/gpt-image-2", json!(0.211)),
            ("fal_ai/high/1024-x-768/openai/gpt-image-2", json!(0.145)),
            ("fal_ai/high/1024-x-1536/openai/gpt-image-2", json!(0.165)),
        ];
        assert_eq!(picked(&keys, "high", target).as_deref(), Some(expected));
    }

    #[rstest]
    fn an_equal_pixel_count_keeps_the_first_row() {
        let keys = [
            ("fal_ai/low/1536-x-1024/openai/gpt-image-2", json!(0.2)),
            ("fal_ai/low/1024-x-1536/openai/gpt-image-2", json!(0.1)),
        ];
        assert_eq!(
            picked(&keys, "low", (1024, 1536)).as_deref(),
            Some("fal_ai/low/1536-x-1024/openai/gpt-image-2")
        );
    }

    #[rstest]
    #[case::other_quality("fal_ai/low/1024-x-1024/openai/gpt-image-2", json!(0.1))]
    #[case::other_model("fal_ai/high/1024-x-1024/openai/gpt-image-2/edit", json!(0.1))]
    #[case::unparsed_size("fal_ai/high/big/openai/gpt-image-2", json!(0.1))]
    #[case::zero_side("fal_ai/high/0-x-1024/openai/gpt-image-2", json!(0.1))]
    #[case::price_not_a_number("fal_ai/high/1024-x-1024/openai/gpt-image-2", json!("0.1"))]
    fn rows_main_skips_are_never_picked(#[case] key: &str, #[case] price: serde_json::Value) {
        assert_eq!(picked(&[(key, price)], "high", (1024, 1024)), None);
    }
}

#[cfg(test)]
mod fal_parity {
    //! Parity with main's fal.ai `cost_calculator` and `fal_ai_passthrough_cost` over rows Model Catalog picked and
    //! lowered. `fal_parity.json` was captured once from main's Python: each case holds the facts the adapter reads off
    //! the call, the deployment's prices, the `fal_ai/<model>` entry and the fallback row, and main's total. The size
    //! keyed rows are the bundled cost map's

    use litellm_cost::query::{
        AdjustmentPolicy, Basis, Candidate, FalCall, FalImage, FalQuery, Header, PriceQuery,
        PriceSource, Provider,
    };
    use litellm_cost::receipt::PriceOutcome;
    use rstest::rstest;
    use serde_json::Value;

    use crate::pricing::priced_ref;
    use crate::test_support::{bundled_rows, names, snapshot};
    use crate::wire::Fields;

    const FIXTURE: &str = include_str!("../fal_parity.json");

    fn case(name: &str) -> Value {
        let fixture: Value = serde_json::from_str(FIXTURE).expect("the fixture is JSON");
        fixture["cases"]
            .as_array()
            .expect("the fixture lists cases")
            .iter()
            .find(|case| case["name"] == name)
            .unwrap_or_else(|| panic!("no fixture case {name}"))
            .clone()
    }

    fn fields(value: &Value) -> Option<Fields> {
        value.as_object().cloned()
    }

    fn priced(case: &Value) -> Option<f64> {
        let snapshot = snapshot(bundled_rows(), names(&["fal_ai"]));
        let model = case["model"].as_str().expect("a model");
        let candidate = Candidate {
            model: model.into(),
            provider: Provider::Other("fal_ai".into()),
        };
        let images = case["images"].as_array().cloned().unwrap_or_default();
        let target = |image: &Value| -> (u64, u64) {
            serde_json::from_value(image["target"].clone()).expect("a target size")
        };
        let keyed: Vec<_> = images
            .iter()
            .map(|image| {
                snapshot.fal_sized_row(
                    model,
                    case["quality"].as_str().unwrap_or("high"),
                    target(image),
                )
            })
            .collect();
        let rows = snapshot.pricing_rows([], keyed.iter().flatten().cloned());
        let rows = match fields(&case["entry"]) {
            Some(entry) => rows.with_named_per_call(candidate.clone(), "catalog", entry, "fal_ai"),
            None => rows,
        };
        let (rows, deployment) = match fields(&case["deployment"]) {
            Some(table) => {
                let (rows, at) = rows.with_per_call("deployment", table, "fal_ai");
                (rows, Some(at))
            }
            None => (rows, None),
        };
        let (rows, fallback) = match fields(&case["fallback"]) {
            Some(table) => {
                let (rows, at) = rows.with_per_call("fallback", table, "fal_ai");
                (rows, Some(at))
            }
            None => (rows, None),
        };
        let header = Header {
            candidates: vec![candidate],
            source: match deployment {
                Some(at) => PriceSource::Rows { rows: vec![at] },
                None => PriceSource::Catalog,
            },
            basis: Basis {
                service_tier: Default::default(),
                data_residency: None,
                vertex_location: None,
                at: jiff::Timestamp::UNIX_EPOCH,
            },
            adjustments: AdjustmentPolicy::None,
            reported: None,
        };
        let text = |key: &str| case[key].as_str().map(Into::into);
        let call = match case["call"].as_str() {
            Some("image") => FalCall::Image {
                images: images
                    .iter()
                    .zip(keyed)
                    .map(|(image, keyed)| FalImage {
                        keyed: keyed.as_ref().map(priced_ref),
                        pixels: image["pixels"].as_u64(),
                    })
                    .collect(),
                billable_units: case["billable_units"].as_f64(),
                resolution: text("resolution"),
                fallback,
            },
            _ => FalCall::Submit {
                seconds: case["seconds"].as_f64(),
                clip_resolution: text("clip_resolution"),
                image_resolution: text("image_resolution"),
                images: case["images"].as_u64().expect("an image count"),
            },
        };
        match litellm_cost::price(&rows, &PriceQuery::Fal(Box::new(FalQuery { header, call }))) {
            PriceOutcome::Priced(receipt) => Some(receipt.total),
            PriceOutcome::Unpriced(_) => None,
        }
    }

    #[rstest]
    #[case::keyed_by_returned_size("keyed_by_returned_size")]
    #[case::keyed_default_size("keyed_default_size")]
    #[case::keyed_named_size_low("keyed_named_size_low")]
    #[case::keyed_custom_size("keyed_custom_size")]
    #[case::entry_resolution("entry_resolution")]
    #[case::entry_billable_units("entry_billable_units")]
    #[case::deployment_per_image("deployment_per_image")]
    #[case::deployment_billable_units("deployment_billable_units")]
    #[case::fallback_megapixels("fallback_megapixels")]
    #[case::fallback_flat("fallback_flat")]
    #[case::fallback_deployment_pixel("fallback_deployment_pixel")]
    #[case::no_images("no_images")]
    #[case::submit_clip("submit_clip")]
    #[case::submit_clip_half_rounds_even("submit_clip_half_rounds_even")]
    #[case::submit_per_video_without_cell("submit_per_video_without_cell")]
    #[case::submit_per_video("submit_per_video")]
    #[case::submit_image_resolution("submit_image_resolution")]
    #[case::submit_image_flat("submit_image_flat")]
    #[case::submit_billable_unit_row("submit_billable_unit_row")]
    fn a_fal_call_costs_what_main_charged(#[case] name: &str) {
        let case = case(name);
        assert_eq!(priced(&case), case["expected"].as_f64());
    }
}
