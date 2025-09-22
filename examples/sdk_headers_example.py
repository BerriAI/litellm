#!/usr/bin/env python3
"""
Example demonstrating LiteLLM SDK header support for enterprise environments.

This example shows how to use additional headers with API gateways, service meshes,
and multi-tenant architectures.
"""

import litellm
import os
from typing import Dict, Any

def example_global_headers():
    """Example: Set global headers for all requests"""
    print("=== Global Headers Example ===")

    # Set global headers that will be included in all API requests
    litellm.headers = {
        "X-API-Gateway-Key": "your-gateway-key-here",
        "X-Company-ID": "acme-corp",
        "X-Environment": "production"
    }

    print("Global headers set:", litellm.headers)

    # These headers will now be included in all completion calls
    # (Note: This example doesn't actually make API calls)
    print("Global headers will be included in all subsequent completion() calls")


def example_per_request_headers():
    """Example: Using extra_headers for specific requests"""
    print("\n=== Per-Request Headers Example ===")

    headers_to_send = {
        "X-Request-ID": "req-12345",
        "X-Tenant-ID": "tenant-abc",
        "X-Custom-Auth": "bearer-token-xyz"
    }

    print("Per-request headers:", headers_to_send)

    # Example of how you would use extra_headers in a real call
    # response = litellm.completion(
    #     model="claude-3-5-sonnet-latest",
    #     messages=[{"role": "user", "content": "Hello"}],
    #     extra_headers=headers_to_send
    # )


def example_header_priority():
    """Example: Demonstrating header priority and merging"""
    print("\n=== Header Priority Example ===")

    # Set global headers
    litellm.headers = {
        "X-Company-ID": "acme-corp",
        "X-Shared-Header": "global-value"
    }

    # Headers that would be sent in a request
    extra_headers = {
        "X-Request-ID": "req-12345",
        "X-Shared-Header": "extra-value"  # Overrides global
    }

    request_headers = {
        "X-Priority-Header": "important",
        "X-Shared-Header": "request-value"  # Overrides both global and extra
    }

    print("Global headers:", litellm.headers)
    print("Extra headers:", extra_headers)
    print("Request headers:", request_headers)
    print("\nFinal headers would be:")
    print("  X-Company-ID: acme-corp (from global)")
    print("  X-Request-ID: req-12345 (from extra)")
    print("  X-Priority-Header: important (from request)")
    print("  X-Shared-Header: request-value (request wins - highest priority)")


def example_enterprise_api_gateway():
    """Example: Enterprise API Gateway scenario"""
    print("\n=== Enterprise API Gateway Example ===")

    # Simulate enterprise environment with Apigee or similar
    gateway_config = {
        "X-API-Gateway-Key": os.getenv("API_GATEWAY_KEY", "demo-key"),
        "X-Route-Version": "v2",
        "X-Rate-Limit-Group": "premium"
    }

    # Set gateway headers globally
    litellm.headers = gateway_config
    print("Gateway headers configured:", gateway_config)

    # Function to make tenant-specific requests
    def make_tenant_request(tenant_id: str, user_id: str, content: str) -> Dict[str, Any]:
        """Make an AI request with tenant-specific headers"""

        tenant_headers = {
            "X-Tenant-ID": tenant_id,
            "X-User-ID": user_id,
            "X-Request-Time": "2024-01-01T00:00:00Z",
            "X-Service-Name": "ai-assistant"
        }

        print(f"Making request for tenant {tenant_id}, user {user_id}")
        print("Tenant-specific headers:", tenant_headers)

        # In a real scenario, this would make the actual API call:
        # return litellm.completion(
        #     model="claude-3-5-sonnet-latest",
        #     messages=[{"role": "user", "content": content}],
        #     extra_headers=tenant_headers
        # )

        # For demo purposes, return mock data
        return {"mock": "response", "headers_used": {**gateway_config, **tenant_headers}}

    # Example usage
    result = make_tenant_request("tenant-123", "user-456", "Analyze this data")
    print("Response:", result)


def example_service_mesh():
    """Example: Service mesh integration (Istio, Linkerd)"""
    print("\n=== Service Mesh Example ===")

    service_mesh_headers = {
        "X-Trace-ID": "trace-abc-123",
        "X-Span-ID": "span-def-456",
        "X-Service-Name": "ai-service",
        "X-Version": "1.2.3",
        "X-Cluster": "prod-us-west-2"
    }

    print("Service mesh headers:", service_mesh_headers)

    # Example of using these headers for distributed tracing
    # response = litellm.completion(
    #     model="gpt-4",
    #     messages=[{"role": "user", "content": "Hello"}],
    #     extra_headers=service_mesh_headers
    # )


def example_debugging_and_monitoring():
    """Example: Request debugging and monitoring"""
    print("\n=== Debugging and Monitoring Example ===")

    import uuid
    import time

    # Generate unique identifiers for request tracking
    trace_id = str(uuid.uuid4())
    request_id = f"req-{int(time.time())}"

    debug_headers = {
        "X-Trace-ID": trace_id,
        "X-Request-ID": request_id,
        "X-Debug-Mode": "true",
        "X-Source-Service": "customer-support-bot",
        "X-Request-Priority": "high"
    }

    print("Debug headers:", debug_headers)
    print(f"Trace ID: {trace_id}")
    print(f"Request ID: {request_id}")

    # These headers help with:
    # 1. Distributed tracing across services
    # 2. Request correlation in logs
    # 3. Debug mode enablement
    # 4. Priority-based routing


if __name__ == "__main__":
    print("LiteLLM SDK Header Support Examples")
    print("=" * 50)

    example_global_headers()
    example_per_request_headers()
    example_header_priority()
    example_enterprise_api_gateway()
    example_service_mesh()
    example_debugging_and_monitoring()

    print("\n" + "=" * 50)
    print("All examples completed!")
    print("\nTo use in your application:")
    print("1. Set litellm.headers for global headers")
    print("2. Use extra_headers parameter for request-specific headers")
    print("3. Use headers parameter for highest priority headers")
    print("4. Headers are merged with priority: headers > extra_headers > litellm.headers")