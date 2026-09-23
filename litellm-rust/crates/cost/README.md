# litellm-cost

This crate calculates token and non-token charges from rates and usage supplied by its caller. It is standalone and has no Python bridge or proxy integration

Call `compile(&pricing)` once for an immutable plan, then `plan.calculate(&request)` for each supported request. `calculate(&pricing, &request)` compiles on each call. A successful result exposes pre-multiplier component costs, selected rates, the multiplier, and derived `input()`, `output()`, and `total()` values

`non_token::calculate` totals priced units such as images, pixels, seconds, pages, requests, credits, and guardrail units. `non_token::calculate_image` selects the first available price from ordered pricing tables, with per-image rates ahead of per-pixel rates inside each table. `non_token::calculate_ocr_with_tables` selects each OCR rate from the first table that has it, then `non_token::calculate_ocr` prefers credit pricing when credits are present and otherwise prices pages and annotation pages. `non_token::calculate_ocr_batch` uses deployment batch or standard rates for each page family, then published rates and the page rate as an annotation fallback. `non_token::calculate_video` selects a video-specific per-second rate, then a resolution rate, then the base per-second rate. Unpriced OCR and video usage returns zero, as in Python. Invalid quantities and rates return errors

`batch::get_batch_cost_rates` selects input, output, cache read, and cache creation rates independently at the highest crossed threshold. `batch::batch_cost_calculator` applies batch modality rates, cache rates, the regular-rate half-price fallback, and an optional regional multiplier to normalized usage. The caller supplies model prices, provider threshold policy, regional multiplier, and token details

`custom_pricing::normalize_cache_usage` applies Python's cache field precedence and adjusts prompt tokens when cache counts use the Anthropic convention. `custom_pricing::cost_per_token_custom_pricing_helper` prices normalized usage with caller-supplied token or per-second rates. The module validates finite nonnegative quantities and rates before calculation

## Python function map

The Rust module tree does not mirror Python's overall cost module tree. The Python entry points also look up models, normalize responses, and choose providers, while this crate starts with caller-supplied prices and usage. These mappings cover the calculation portions of the functions:

| Python function | Rust function | Python tests | Rust tests |
| --- | --- | --- | --- |
| `litellm.litellm_core_utils.llm_cost_calc.utils.generic_cost_per_token` | `litellm_cost::calculate` | `test_llm_cost_calc_utils.py` | `calculation.rs`, `python_reference.tsv` |
| `litellm.cost_calculator.default_image_cost_calculator` | `litellm_cost::non_token::calculate_image` | `test_cost_calculator.py::test_default_image_cost_calculator` | `python_cost_calculator.rs::default_image_cost_calculator_selects_first_priced_unit` |
| `litellm.cost_calculator.ocr_cost` | `litellm_cost::non_token::calculate_ocr_with_tables` | `test_cost_calculator.py::test_ocr_cost_*` | `python_cost_calculator.rs::ocr_cost_*` |
| `litellm.cost_calculator.ocr_batch_cost` | `litellm_cost::non_token::calculate_ocr_batch` | `litellm_core_utils/test_litellm_logging.py::test_ocr_only_deployment_pricing_reaches_batch_ocr_cost` | `python_cost_calculator.rs::ocr_batch_cost_*` |
| `litellm.cost_calculator.default_video_cost_calculator` | `litellm_cost::non_token::calculate_video` | `test_video_generation.py::test_video_generation_cost_*` | `python_cost_calculator.rs::default_video_cost_calculator_*` |
| `litellm.litellm_core_utils.llm_cost_calc.utils.get_batch_cost_rates` | `litellm_cost::batch::get_batch_cost_rates` | `test_cost_calculator.py::test_batch_cost_calculator_*` | `python_cost_calculator.rs::get_batch_cost_rates_*` |
| `litellm.cost_calculator.batch_cost_calculator` | `litellm_cost::batch::batch_cost_calculator` | `test_cost_calculator.py::test_batch_cost_calculator_*` | `python_cost_calculator.rs::batch_cost_calculator_*` |
| `litellm.cost_calculator.cost_per_token` cache normalization | `litellm_cost::custom_pricing::normalize_cache_usage` | `test_cost_calculator.py::test_custom_pricing_*` | `python_custom_pricing.rs::normalize_cache_usage_*` |
| `litellm.cost_calculator._cost_per_token_custom_pricing_helper` | `litellm_cost::custom_pricing::cost_per_token_custom_pricing_helper` | `test_cost_calculator.py::test_custom_pricing_*` | `python_custom_pricing.rs::cost_per_token_custom_pricing_helper_*` |

`cost_per_token`, `completion_cost`, `response_cost_calculator`, provider calculators, and Python's model lookup and response normalization have no Rust counterpart yet. The ported Rust cases use `rstest` and synthetic prices; they cover the corresponding Python tests' price selection and arithmetic, not their integration with Python model registration

The caller states whether `prompt_tokens` includes cache tokens. Threshold selection uses total input tokens for either convention and selects one rate for the whole request. Thresholds are sorted when compiled, and duplicate thresholds or tier overrides fail deterministically. `Fast` selects priority rates; unknown tiers use standard rates

`Rate::Missing`, `Rate::Null`, and `Rate::Value(0.0)` remain distinct. Missing cache rates fall back to the selected input rate, and an absent one-hour write rate falls back to the selected write rate. Missing input or output rates return typed errors, including for zero usage. Python's sparse-entry behavior remains outside this native contract

The supported off-peak shape is one non-wrapping UTC daily window. The caller supplies the applicable regional multiplier after provider-specific selection. Negative or non-finite rates, ambiguous rules, inconsistent cache counts, incomplete write splits, invalid windows and overflow return errors. Callers must decline unsupported inputs before native execution if their public contract accepts those shapes

This crate does not select models, read catalogs, fetch provider prices, normalize provider responses, or process provider-reported costs. The non-token functions accept already selected prices and do not yet match every Python fallback or warning. This crate does not change proxy behavior. The reference fixture was generated by `tests/generate_python_reference.py` against the Python implementation at the commit recorded in `tests/python_reference.tsv`, using synthetic rates and fixed usage
