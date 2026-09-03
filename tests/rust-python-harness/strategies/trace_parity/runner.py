"""Trace parity strategy runner.

Validates Python-to-Rust mapping by tracing actual execution.
"""
import sys
from pathlib import Path


def run_gateway_validation():
    """Run all gateway trace validations."""
    gateway_dir = Path(__file__).parent / "gateway"

    print("=== Gateway Trace Parity Validation ===\n")

    print("Running core validation...")
    from gateway.validate_core import validate_chat_completions
    validate_chat_completions()

    print("\nRunning ai-gateway validation...")
    from gateway.validate_ai_gateway import validate_ocr
    validate_ocr()

    print("\n✅ All gateway validations complete")


if __name__ == "__main__":
    run_gateway_validation()
