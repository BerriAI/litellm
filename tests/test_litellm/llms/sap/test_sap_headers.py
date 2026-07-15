from litellm.llms.sap.headers import merge_sap_request_headers


def test_merge_sap_request_headers_forwards_custom_headers():
    result = merge_sap_request_headers(
        provider_headers={
            "Authorization": "Bearer SAP_TOKEN",
            "AI-Resource-Group": "default",
            "Content-Type": "application/json",
            "AI-Client-Type": "LiteLLM",
        },
        caller_headers={
            "ai-inference-observability-persistence-mode": "all",
        },
    )

    assert result["Authorization"] == "Bearer SAP_TOKEN"
    assert result["ai-inference-observability-persistence-mode"] == "all"


def test_merge_sap_request_headers_strips_reserved_headers_case_insensitively():
    result = merge_sap_request_headers(
        provider_headers={
            "Authorization": "Bearer SAP_TOKEN",
            "AI-Resource-Group": "default",
            "Content-Type": "application/json",
            "AI-Client-Type": "LiteLLM",
        },
        caller_headers={
            "authorization": "Bearer PROXY_TOKEN",
            "AI-RESOURCE-GROUP": "spoofed-group",
            "Content-Type": "application/xml",
            "ai-client-type": "spoofed-client",
            "ai-inference-observability-persistence-mode": "all",
        },
    )

    assert result["Authorization"] == "Bearer SAP_TOKEN"
    assert result["AI-Resource-Group"] == "default"
    assert result["Content-Type"] == "application/json"
    assert result["AI-Client-Type"] == "LiteLLM"
    assert result["ai-inference-observability-persistence-mode"] == "all"
    assert "spoofed-group" not in result.values()
    assert "Bearer PROXY_TOKEN" not in result.values()
