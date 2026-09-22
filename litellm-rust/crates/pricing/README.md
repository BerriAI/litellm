# litellm-pricing

A standalone, pure Rust calculator for text token charges. The caller supplies resolved per-token USD rates, normalized usage, service tier, threshold policy, optional regional multiplier, and UTC minute of day when off-peak pricing is configured

`Rate::Missing`, `Rate::Null`, and `Rate::Value(0.0)` remain distinct. Missing and null cache rates fall back to the selected input rate; zero is free. Missing input or output rates are errors even for zero usage. Prompt counts may include cached tokens or exclude them, as declared by `PromptConvention`. Cache writes may provide a complete 5-minute and 1-hour split. Thresholds choose one rate for the whole request, with at most one tier-specific override per threshold; `Fast` uses priority rates. Off-peak rates replace selected rates within one non-wrapping UTC daily interval. A regional multiplier applies to both totals

Callers must resolve model names, provider defaults, deployment overrides, and raw usage before calling this crate. It does not price audio, images, video, reasoning breakdowns, characters, tools, batch jobs, multiple off-peak windows, tiered tables, discounts, margins, or persistence. Inconsistent cache accounting and incomplete write splits return errors. The calculator uses `f64` multiplication and addition in the same component order as the supported Python cost paths; it does not round to decimal currency

The tests use synthetic prices and cover Python's ordinary text/cache arithmetic, custom-price cache fallback, cache-write duration, threshold and service-tier boundaries. They do not pin vendor catalog prices
