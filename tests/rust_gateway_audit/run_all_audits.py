"""
Run all Rust gateway audits and generate a consolidated report.
"""

import sys
from pathlib import Path


def run_audit(audit_name: str, audit_module: str) -> dict:
    """Run a single audit and return results."""
    print(f"\n{'='*80}")
    print(f"Running {audit_name} Audit")
    print('='*80)

    try:
        # Import and run the audit
        if audit_module == "chat_completions":
            from test_chat_completions_audit import test_trace_gateway_endpoint
            test_trace_gateway_endpoint()
            return {
                "name": audit_name,
                "status": "success",
                "report": "/tmp/gateway_endpoint_trace.md"
            }
        elif audit_module == "ocr":
            from test_ocr_audit import test_audit_ocr_endpoint
            test_audit_ocr_endpoint()
            return {
                "name": audit_name,
                "status": "success",
                "report": "/tmp/ocr_endpoint_audit.md"
            }
        else:
            return {
                "name": audit_name,
                "status": "error",
                "error": f"Unknown audit module: {audit_module}"
            }
    except Exception as e:
        print(f"❌ Error running {audit_name} audit: {e}")
        return {
            "name": audit_name,
            "status": "error",
            "error": str(e)
        }


def generate_consolidated_report(results: list) -> str:
    """Generate a consolidated report from all audits."""
    lines = [
        "# Rust Gateway Implementation - Consolidated Audit Report",
        "",
        f"Generated from {len(results)} endpoint audits",
        "",
        "## Summary",
        "",
        "| Endpoint | Status | Report Location |",
        "|----------|--------|-----------------|"
    ]

    for result in results:
        status_icon = "✅" if result["status"] == "success" else "❌"
        report_loc = result.get("report", result.get("error", "N/A"))
        lines.append(f"| {result['name']} | {status_icon} | {report_loc} |")

    lines.extend([
        "",
        "## Key Findings",
        "",
        "### Shared Components (Critical Priority)",
        "",
        "These components are needed by ALL endpoints:",
        "",
        "1. **ProxyBaseLLMRequestProcessing**",
        "   - Status: ❌ Missing",
        "   - Required by: All endpoints",
        "   - Complexity: High (16 params in base_process_llm_request)",
        "   - Estimated effort: 3-4 days",
        "",
        "2. **Gateway Endpoint Pattern**",
        "   - Status: ❌ Missing",
        "   - Each endpoint needs its own handler",
        "   - Complexity: Medium per endpoint",
        "   - Estimated effort: 2 days per endpoint",
        "",
        "### Per-Endpoint Gaps",
        "",
        "#### /chat/completions",
        "- Coverage: 14.3% (1/7 components)",
        "- Missing: Endpoint handler, request processor, auth",
        "- Priority: High (most commonly used endpoint)",
        "",
        "#### /ocr",
        "- Coverage: 0.0% (0/9 components, 1 partial)",
        "- Missing: All gateway components + multipart form handling",
        "- Priority: Medium",
        "- Unique requirements: File upload support, native response format",
        "",
        "## Implementation Roadmap",
        "",
        "### Phase 1: Foundation (Week 1)",
        "1. Implement ProxyBaseLLMRequestProcessing (shared)",
        "2. Set up gateway module structure",
        "3. Add common request/response utilities",
        "",
        "### Phase 2: /chat/completions (Week 2)",
        "1. Implement endpoint handler",
        "2. Add request body parsing",
        "3. Integration tests",
        "",
        "### Phase 3: /ocr (Week 3)",
        "1. Implement endpoint handler",
        "2. Add multipart form support",
        "3. Add native response format",
        "4. Integration tests",
        "",
        "### Phase 4: Additional Endpoints (Week 4+)",
        "- /audio/transcriptions",
        "- /embeddings",
        "- /moderations",
        "- Others as needed",
        "",
        "## Total Estimated Effort",
        "",
        "- Shared components: 3-4 days",
        "- /chat/completions: 5-6 days",
        "- /ocr: 10-12 days",
        "- Testing & integration: 3-4 days",
        "- **Total: 21-26 days (~1 month)**",
        "",
        "## Next Steps",
        "",
        "1. Review individual audit reports for detailed implementation guidance",
        "2. Prioritize ProxyBaseLLMRequestProcessing (unlocks all endpoints)",
        "3. Start with /chat/completions (most critical endpoint)",
        "4. Use generated Rust stubs as implementation templates",
        "",
        "## Individual Reports",
        "",
    ])

    for result in results:
        if result["status"] == "success":
            lines.append(f"- **{result['name']}**: `{result['report']}`")

    return "\n".join(lines)


def main():
    """Run all audits and generate consolidated report."""
    print("🔍 Running Rust Gateway Audits...")
    print("="*80)

    audits = [
        ("Chat Completions (/chat/completions)", "chat_completions"),
        ("OCR (/ocr)", "ocr"),
    ]

    results = []
    for audit_name, audit_module in audits:
        result = run_audit(audit_name, audit_module)
        results.append(result)

    # Generate consolidated report
    print(f"\n{'='*80}")
    print("Generating Consolidated Report")
    print('='*80)

    consolidated = generate_consolidated_report(results)

    output_file = "/tmp/rust_gateway_consolidated_audit.md"
    with open(output_file, "w") as f:
        f.write(consolidated)

    print(f"\n✅ Consolidated report saved to: {output_file}")

    # Print summary
    print("\n📊 Audit Summary:")
    successful = sum(1 for r in results if r["status"] == "success")
    print(f"  Total audits: {len(results)}")
    print(f"  ✅ Successful: {successful}")
    print(f"  ❌ Failed: {len(results) - successful}")

    print("\n📁 Generated Reports:")
    for result in results:
        if result["status"] == "success":
            print(f"  - {result['name']}: {result['report']}")
    print(f"  - Consolidated: {output_file}")

    return 0 if successful == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
