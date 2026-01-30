#!/usr/bin/env python3
"""
Comprehensive validation script for model_prices_and_context_window.json

This script validates the model cost map to prevent malformed entries from
breaking LiteLLM deployments. Run this in CI before merging any changes to
the model cost map.

Usage:
    python scripts/validate_model_cost_map.py [path_to_json]
    
If no path is provided, defaults to model_prices_and_context_window.json
in the repository root.

Exit codes:
    0 - Validation passed
    1 - Validation failed
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# Validation configuration
MIN_MODEL_ENTRIES = 100
MAX_COST_PER_TOKEN = 1.0  # $1 per token would be absurdly high

VALID_MODES = {
    "audio_speech",
    "audio_transcription",
    "chat",
    "completion",
    "embedding",
    "image_edit",
    "image_generation",
    "moderation",
    "ocr",
    "rerank",
    "responses",
    "search",
    "vector_store",
    "video_generation",
}

# Fields that should be numeric if present
NUMERIC_FIELDS = {
    "input_cost_per_token",
    "output_cost_per_token",
    "input_cost_per_character",
    "output_cost_per_character",
    "input_cost_per_image",
    "output_cost_per_image",
    "input_cost_per_audio_token",
    "output_cost_per_audio_token",
    "input_cost_per_second",
    "output_cost_per_second",
    "max_tokens",
    "max_input_tokens",
    "max_output_tokens",
}

# Fields that should be boolean if present
BOOLEAN_FIELDS = {
    "supports_function_calling",
    "supports_parallel_function_calling",
    "supports_vision",
    "supports_audio_input",
    "supports_audio_output",
    "supports_prompt_caching",
    "supports_response_schema",
    "supports_system_messages",
    "supports_reasoning",
    "supports_web_search",
}


class ValidationError:
    def __init__(self, model: str, field: str, message: str, severity: str = "error"):
        self.model = model
        self.field = field
        self.message = message
        self.severity = severity  # "error" or "warning"
    
    def __str__(self):
        return f"[{self.severity.upper()}] {self.model}: {self.field} - {self.message}"


def validate_entry(model_name: str, entry: Any) -> List[ValidationError]:
    """Validate a single model entry."""
    errors = []
    
    # Skip sample_spec - it's documentation
    if model_name == "sample_spec":
        return errors
    
    # Entry must be a dict
    if not isinstance(entry, dict):
        errors.append(ValidationError(
            model_name, "entry", f"Expected dict, got {type(entry).__name__}"
        ))
        return errors
    
    # Required field: litellm_provider
    if "litellm_provider" not in entry:
        errors.append(ValidationError(
            model_name, "litellm_provider", "Missing required field"
        ))
    elif not isinstance(entry["litellm_provider"], str):
        errors.append(ValidationError(
            model_name, "litellm_provider", 
            f"Expected string, got {type(entry['litellm_provider']).__name__}"
        ))
    elif not entry["litellm_provider"].strip():
        errors.append(ValidationError(
            model_name, "litellm_provider", "Cannot be empty string"
        ))
    
    # Validate mode if present
    if "mode" in entry:
        if not isinstance(entry["mode"], str):
            errors.append(ValidationError(
                model_name, "mode", 
                f"Expected string, got {type(entry['mode']).__name__}"
            ))
        elif entry["mode"] not in VALID_MODES:
            errors.append(ValidationError(
                model_name, "mode", 
                f"Invalid mode '{entry['mode']}'. Valid modes: {sorted(VALID_MODES)}"
            ))
    
    # Validate numeric fields
    for field in NUMERIC_FIELDS:
        if field in entry:
            value = entry[field]
            if isinstance(value, str):
                # Some fields in sample_spec have string descriptions
                if model_name != "sample_spec":
                    errors.append(ValidationError(
                        model_name, field,
                        f"Expected numeric, got string: '{value}'"
                    ))
            elif not isinstance(value, (int, float)):
                errors.append(ValidationError(
                    model_name, field,
                    f"Expected numeric, got {type(value).__name__}"
                ))
            elif isinstance(value, (int, float)) and value < 0:
                errors.append(ValidationError(
                    model_name, field,
                    f"Cannot be negative: {value}",
                    severity="warning"
                ))
    
    # Validate boolean fields
    for field in BOOLEAN_FIELDS:
        if field in entry:
            value = entry[field]
            if not isinstance(value, bool):
                errors.append(ValidationError(
                    model_name, field,
                    f"Expected boolean, got {type(value).__name__}: {value}"
                ))
    
    # Validate cost sanity (catch obviously wrong values)
    for cost_field in ["input_cost_per_token", "output_cost_per_token"]:
        if cost_field in entry:
            value = entry[cost_field]
            if isinstance(value, (int, float)) and value > MAX_COST_PER_TOKEN:
                errors.append(ValidationError(
                    model_name, cost_field,
                    f"Suspiciously high cost: ${value}/token (max expected: ${MAX_COST_PER_TOKEN})",
                    severity="warning"
                ))
    
    # Validate max_tokens relationships
    max_tokens = entry.get("max_tokens")
    max_input = entry.get("max_input_tokens")
    max_output = entry.get("max_output_tokens")
    
    if max_input is not None and max_output is not None:
        if isinstance(max_input, (int, float)) and isinstance(max_output, (int, float)):
            if max_input < 0 or max_output < 0:
                pass  # Already caught above
            elif max_input == 0 and max_output == 0:
                errors.append(ValidationError(
                    model_name, "max_tokens",
                    "Both max_input_tokens and max_output_tokens are 0",
                    severity="warning"
                ))
    
    return errors


def validate_model_cost_map(data: Dict[str, Any]) -> Tuple[List[ValidationError], Dict[str, int]]:
    """
    Validate the entire model cost map.
    
    Returns:
        Tuple of (errors, stats)
    """
    all_errors = []
    stats = {
        "total_entries": 0,
        "valid_entries": 0,
        "entries_with_errors": 0,
        "entries_with_warnings": 0,
        "total_errors": 0,
        "total_warnings": 0,
    }
    
    if not isinstance(data, dict):
        all_errors.append(ValidationError(
            "ROOT", "type", f"Expected dict at root, got {type(data).__name__}"
        ))
        return all_errors, stats
    
    # Count entries (excluding sample_spec)
    model_entries = {k: v for k, v in data.items() if k != "sample_spec"}
    stats["total_entries"] = len(model_entries)
    
    # Check minimum entries
    if stats["total_entries"] < MIN_MODEL_ENTRIES:
        all_errors.append(ValidationError(
            "ROOT", "count",
            f"Too few model entries: {stats['total_entries']} (minimum: {MIN_MODEL_ENTRIES})"
        ))
    
    # Validate each entry
    for model_name, entry in data.items():
        errors = validate_entry(model_name, entry)
        
        entry_errors = [e for e in errors if e.severity == "error"]
        entry_warnings = [e for e in errors if e.severity == "warning"]
        
        if entry_errors:
            stats["entries_with_errors"] += 1
        elif entry_warnings:
            stats["entries_with_warnings"] += 1
        else:
            stats["valid_entries"] += 1
        
        stats["total_errors"] += len(entry_errors)
        stats["total_warnings"] += len(entry_warnings)
        
        all_errors.extend(errors)
    
    return all_errors, stats


def main():
    # Determine file path
    if len(sys.argv) > 1:
        json_path = Path(sys.argv[1])
    else:
        # Default to repository root
        script_dir = Path(__file__).parent
        json_path = script_dir.parent / "model_prices_and_context_window.json"
    
    if not json_path.exists():
        print(f"ERROR: File not found: {json_path}")
        sys.exit(1)
    
    print(f"Validating: {json_path}")
    print("-" * 60)
    
    # Load and parse JSON
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"ERROR: Invalid JSON: {e}")
        sys.exit(1)
    
    # Validate
    errors, stats = validate_model_cost_map(data)
    
    # Print statistics
    print(f"Total model entries: {stats['total_entries']}")
    print(f"Valid entries: {stats['valid_entries']}")
    print(f"Entries with errors: {stats['entries_with_errors']}")
    print(f"Entries with warnings: {stats['entries_with_warnings']}")
    print("-" * 60)
    
    # Print errors (errors first, then warnings)
    error_list = [e for e in errors if e.severity == "error"]
    warning_list = [e for e in errors if e.severity == "warning"]
    
    if error_list:
        print(f"\nERRORS ({len(error_list)}):")
        for error in error_list[:50]:  # Limit output
            print(f"  {error}")
        if len(error_list) > 50:
            print(f"  ... and {len(error_list) - 50} more errors")
    
    if warning_list:
        print(f"\nWARNINGS ({len(warning_list)}):")
        for warning in warning_list[:20]:  # Limit output
            print(f"  {warning}")
        if len(warning_list) > 20:
            print(f"  ... and {len(warning_list) - 20} more warnings")
    
    # Exit with appropriate code
    if error_list:
        print(f"\n❌ VALIDATION FAILED: {len(error_list)} error(s) found")
        sys.exit(1)
    elif warning_list:
        print(f"\n⚠️  VALIDATION PASSED with {len(warning_list)} warning(s)")
        sys.exit(0)
    else:
        print("\n✅ VALIDATION PASSED")
        sys.exit(0)


if __name__ == "__main__":
    main()
