from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import jsonschema

REPO_ROOT = Path(__file__).parent.parent
PRICES_PATH = REPO_ROOT / "model_prices_and_context_window.json"
SCHEMA_PATH = REPO_ROOT / "model_prices_and_context_window.schema.json"

SPECIAL_ROOT_KEYS = frozenset({"sample_spec", "fallback_generalizations"})

JsonSchema = dict

NONNEG_NUMBER: JsonSchema = {"type": "number", "minimum": 0}
NONNEG_INTEGER: JsonSchema = {"type": "integer", "minimum": 0}
BOOLEAN: JsonSchema = {"type": "boolean"}
STRING: JsonSchema = {"type": "string"}

EXTRA_BOOLEAN_KEYS = frozenset(
    {
        "gemini_native_audio",
        "gemini_audio_only_live",
        "uses_embed_content",
        "use_openai_responses_path",
        "bedrock_converse_supports_strict_tools",
    }
)

OBJECT_KEYS: dict[str, JsonSchema] = {
    "search_context_cost_per_query": {
        "type": "object",
        "description": "USD cost per web search query, keyed by search context size.",
        "properties": {
            "search_context_size_low": NONNEG_NUMBER,
            "search_context_size_medium": NONNEG_NUMBER,
            "search_context_size_high": NONNEG_NUMBER,
        },
        "additionalProperties": False,
    },
    "metadata": {
        "type": "object",
        "description": "Free-form notes about the entry (e.g. pricing derivation).",
    },
    "provider_specific_entry": {
        "type": "object",
        "description": "Provider-internal routing hints (e.g. bedrock_invocation_schema).",
    },
}

ARRAY_KEYS: dict[str, JsonSchema] = {
    "supported_endpoints": {
        "type": "array",
        "description": "OpenAI-style API routes this model can be called through, e.g. /v1/chat/completions.",
        "items": STRING,
    },
    "supported_modalities": {
        "type": "array",
        "description": "Input modalities the model accepts.",
        "items": {"type": "string", "enum": ["text", "image", "audio", "video"]},
    },
    "supported_output_modalities": {
        "type": "array",
        "description": "Output modalities the model can produce.",
        "items": {"type": "string", "enum": ["text", "image", "audio", "video", "code"]},
    },
    "supported_regions": {
        "type": "array",
        "description": "Cloud regions the model is available in ('global' or region ids).",
        "items": STRING,
    },
    "tiered_pricing": {
        "type": "array",
        "description": "Context-length or result-count tiered rates; each tier's costs apply within its range.",
        "items": {
            "type": "object",
            "properties": {
                "range": {
                    "type": "array",
                    "description": "[min, max] prompt-token span this tier applies to.",
                    "items": NONNEG_NUMBER,
                    "minItems": 2,
                    "maxItems": 2,
                },
                "max_results_range": {
                    "type": "array",
                    "description": "[min, max] result-count span this tier applies to (search models).",
                    "items": NONNEG_NUMBER,
                    "minItems": 2,
                    "maxItems": 2,
                },
                "input_cost_per_token": NONNEG_NUMBER,
                "output_cost_per_token": NONNEG_NUMBER,
                "output_cost_per_reasoning_token": NONNEG_NUMBER,
                "cache_read_input_token_cost": NONNEG_NUMBER,
                "input_cost_per_query": NONNEG_NUMBER,
            },
            "additionalProperties": False,
        },
    },
}

INTEGER_KEYS: dict[str, JsonSchema] = {
    "max_tokens": {
        **NONNEG_INTEGER,
        "description": "Legacy field: max output tokens if the provider specifies it, else max input tokens.",
    },
    "max_input_tokens": {
        **NONNEG_INTEGER,
        "description": "Maximum prompt/context tokens the model accepts.",
    },
    "max_output_tokens": {
        **NONNEG_INTEGER,
        "description": "Maximum tokens the model can generate in one response.",
    },
    "output_vector_size": {
        **NONNEG_INTEGER,
        "description": "Embedding dimension for embedding models.",
    },
    "prompt_cache_min_tokens": {
        **NONNEG_INTEGER,
        "description": "Smallest prefix the provider will actually cache; absent means the provider default applies.",
    },
    "tpm": {**NONNEG_INTEGER, "description": "Provider default tokens-per-minute limit."},
    "rpm": {**NONNEG_INTEGER, "description": "Provider default requests-per-minute limit."},
}

NUMBER_KEYS: dict[str, JsonSchema] = {
    "regional_processing_uplift_multiplier_eu": {
        "type": "number",
        "minimum": 1,
        "description": "Multiplier applied to all token costs for EU data residency (e.g. 1.10 = +10%).",
    },
    "regional_processing_uplift_multiplier_us": {
        "type": "number",
        "minimum": 1,
        "description": "Multiplier applied to all token costs for US data residency (e.g. 1.10 = +10%).",
    },
}

COST_DESCRIPTIONS: dict[str, str] = {
    "input_cost_per_token": "USD per prompt token.",
    "output_cost_per_token": "USD per generated token.",
    "output_cost_per_reasoning_token": "USD per reasoning/thinking token, when billed separately.",
    "cache_creation_input_token_cost": "USD per token written to the provider's prompt cache.",
    "cache_read_input_token_cost": "USD per prompt token served from the provider's prompt cache.",
    "input_cost_per_token_batches": "USD per prompt token via the provider's batch API.",
    "output_cost_per_token_batches": "USD per generated token via the provider's batch API.",
}


def cost_description(key: str) -> Optional[str]:
    if key in COST_DESCRIPTIONS:
        return COST_DESCRIPTIONS[key]
    if key.endswith("_flex"):
        return "Flex service-tier rate for the same-named base field."
    if key.endswith("_priority"):
        return "Priority service-tier rate for the same-named base field."
    if "_above_" in key:
        return "Rate applied once the prompt exceeds the token threshold in the field name."
    return None


def cost_schema(key: str) -> JsonSchema:
    description = cost_description(key)
    return {**NONNEG_NUMBER, "description": description} if description else dict(NONNEG_NUMBER)


def string_key_schemas(modes: tuple) -> dict[str, JsonSchema]:
    return {
        "litellm_provider": {
            "type": "string",
            "description": "LiteLLM provider slug; one of https://docs.litellm.ai/docs/providers.",
        },
        "mode": {
            "type": "string",
            "description": "Primary API surface / task type of the model.",
            "enum": list(modes),
        },
        "source": {
            "type": "string",
            "description": "URL of the provider pricing/model page this entry was taken from.",
        },
        "deprecation_date": {
            "type": "string",
            "description": "Date the provider deprecates the model, YYYY-MM-DD.",
            "format": "date",
            "pattern": "^\\d{4}-(0[1-9]|1[0-2])-(0[1-9]|[12]\\d|3[01])$",
        },
        "web_search_billing_unit": {
            "type": "string",
            "description": "Whether web search is billed per query or per prompt.",
            "enum": ["per_query", "per_prompt"],
        },
        "bedrock_output_config_effort_ceiling": {
            "type": "string",
            "description": "Highest reasoning effort the Bedrock output_config accepts for this model.",
            "enum": ["low", "medium", "high", "max", "xhigh"],
        },
        "comment": STRING,
        "audio_transcription_config": STRING,
    }


def classify(key: str, modes: tuple) -> Optional[JsonSchema]:
    curated = {**OBJECT_KEYS, **ARRAY_KEYS, **string_key_schemas(modes), **INTEGER_KEYS, **NUMBER_KEYS}
    if key in curated:
        return curated[key]
    if key.startswith("supports_") or key in EXTRA_BOOLEAN_KEYS:
        return BOOLEAN
    if "cost" in key:
        return cost_schema(key)
    return None


def build_schema(prices: dict) -> JsonSchema:
    entries = {name: entry for name, entry in prices.items() if name not in SPECIAL_ROOT_KEYS}
    all_keys = tuple(sorted({key for entry in entries.values() for key in entry}))
    modes = tuple(sorted({entry["mode"] for entry in entries.values() if "mode" in entry}))
    unclassified = tuple(key for key in all_keys if classify(key, modes) is None)
    if unclassified:
        raise SystemExit(
            f"Unclassified keys in {PRICES_PATH.name}: {', '.join(unclassified)}. "
            f"Add them to the key tables in {Path(__file__).name} and rerun it."
        )
    entry_properties = {key: classify(key, modes) for key in all_keys}
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "LiteLLM model_prices_and_context_window.json",
        "description": (
            "Schema for LiteLLM's model price and context window registry "
            "(https://github.com/BerriAI/litellm/blob/main/model_prices_and_context_window.json). "
            "Every top-level key except 'sample_spec' and 'fallback_generalizations' is a model id, "
            "optionally prefixed with its provider (e.g. 'azure/gpt-5.4'), mapping to a model entry. "
            "All costs are USD per unit. New optional fields are added regularly, so consumers should "
            "ignore unknown fields rather than reject them."
        ),
        "type": "object",
        "properties": {
            "sample_spec": {
                "type": "object",
                "description": (
                    "Documentation placeholder illustrating the entry shape; not a real model and not "
                    "schema-conformant (several values are prose)."
                ),
            },
            "fallback_generalizations": {
                "type": "object",
                "description": "Regex rules that generalize unknown model ids to known families; not a model entry.",
                "properties": {
                    "rules": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": STRING,
                                "pattern": STRING,
                                "description": STRING,
                            },
                            "required": ["name", "pattern"],
                            "additionalProperties": True,
                        },
                    }
                },
                "additionalProperties": False,
            },
        },
        "additionalProperties": {"$ref": "#/$defs/modelEntry"},
        "$defs": {
            "modelEntry": {
                "type": "object",
                "description": (
                    "Pricing, limits, and capability flags for one model. Fields other than litellm_provider "
                    "are optional; boolean capability flags are simply omitted when unknown or false."
                ),
                "required": ["litellm_provider"],
                "properties": entry_properties,
                "additionalProperties": True,
            }
        },
    }


def render(schema: JsonSchema) -> str:
    return json.dumps(schema, indent=2) + "\n"


def validation_errors(prices: dict, schema: JsonSchema) -> tuple:
    validator = jsonschema.Draft202012Validator(
        schema, format_checker=jsonschema.Draft202012Validator.FORMAT_CHECKER
    )
    return tuple(
        f"{'.'.join(str(part) for part in error.absolute_path)}: {error.message}"
        for error in validator.iter_errors(prices)
    )


def main() -> int:
    check = "--check" in sys.argv[1:]
    prices = json.loads(PRICES_PATH.read_text())
    rendered = render(build_schema(prices))
    errors = validation_errors(prices, json.loads(rendered))
    if errors:
        print(f"{PRICES_PATH.name} does not validate against the generated schema:")
        print("\n".join(errors[:20]))
        return 1
    if not check:
        SCHEMA_PATH.write_text(rendered)
        print(f"wrote {SCHEMA_PATH}")
        return 0
    if not SCHEMA_PATH.exists() or SCHEMA_PATH.read_text() != rendered:
        print(
            f"{SCHEMA_PATH.name} is out of sync with {PRICES_PATH.name}. "
            f"Run `python {Path(__file__).relative_to(REPO_ROOT)}` and commit the result."
        )
        return 1
    print(f"{SCHEMA_PATH.name} is in sync and {PRICES_PATH.name} validates against it")
    return 0


if __name__ == "__main__":
    sys.exit(main())
