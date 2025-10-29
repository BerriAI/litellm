#!/usr/bin/env python3
"""
Test script to verify X-Forwarded-* headers work correctly.

This simulates what your nginx proxy does by making requests to LiteLLM
with X-Forwarded-* headers set.

Usage:
1. Start your LiteLLM proxy on port 4000 (or whatever port you use)
2. Run this script: python test_proxy_forwarding.py
3. Check the output to verify URLs are correctly constructed
"""

import httpx
import json
from typing import Optional


def test_oauth_endpoints(
    litellm_url: str = "http://localhost:4000",
    external_host: str = "proxy.example.com",
    external_scheme: str = "https",
    mcp_server_name: str = "github",
):
    """
    Test that OAuth discovery endpoints return correct URLs when X-Forwarded-* headers are set.

    Args:
        litellm_url: Internal LiteLLM URL (where it's actually running)
        external_host: External hostname that users see
        external_scheme: External scheme (usually https)
        mcp_server_name: Name of your MCP server
    """

    # Headers that your proxy would set
    forwarded_headers = {
        "X-Forwarded-Proto": external_scheme,
        "X-Forwarded-Host": external_host,
    }

    print("=" * 70)
    print("Testing X-Forwarded-* Header Support")
    print("=" * 70)
    print(f"\nInternal URL: {litellm_url}")
    print(f"External URL: {external_scheme}://{external_host}")
    print(f"MCP Server: {mcp_server_name}")
    print(f"\nForwarded Headers: {json.dumps(forwarded_headers, indent=2)}")
    print("\n" + "=" * 70)

    with httpx.Client(timeout=10.0) as client:

        # Test 1: OAuth Authorization Server Discovery
        print("\n[Test 1] OAuth Authorization Server Discovery")
        print("-" * 70)

        url = f"{litellm_url}/.well-known/oauth-authorization-server/{mcp_server_name}"
        print(f"Request: GET {url}")
        print(f"Headers: {json.dumps(forwarded_headers, indent=2)}")

        try:
            response = client.get(url, headers=forwarded_headers)
            print(f"Status: {response.status_code}")

            if response.status_code == 200:
                data = response.json()
                print(f"\nResponse:")
                print(json.dumps(data, indent=2))

                # Verify URLs use external host
                auth_endpoint = data.get("authorization_endpoint", "")
                token_endpoint = data.get("token_endpoint", "")
                reg_endpoint = data.get("registration_endpoint", "")

                print(f"\n✓ Checking endpoints use external URL:")
                expected_prefix = f"{external_scheme}://{external_host}"

                if auth_endpoint.startswith(expected_prefix):
                    print(f"  ✓ authorization_endpoint: {auth_endpoint}")
                else:
                    print(f"  ✗ authorization_endpoint: {auth_endpoint} (expected to start with {expected_prefix})")

                if token_endpoint.startswith(expected_prefix):
                    print(f"  ✓ token_endpoint: {token_endpoint}")
                else:
                    print(f"  ✗ token_endpoint: {token_endpoint} (expected to start with {expected_prefix})")

                if reg_endpoint.startswith(expected_prefix):
                    print(f"  ✓ registration_endpoint: {reg_endpoint}")
                else:
                    print(f"  ✗ registration_endpoint: {reg_endpoint} (expected to start with {expected_prefix})")
            else:
                print(f"Error: {response.text}")

        except Exception as e:
            print(f"Error: {e}")

        # Test 2: OAuth Protected Resource Discovery
        print("\n\n[Test 2] OAuth Protected Resource Discovery")
        print("-" * 70)

        url = f"{litellm_url}/.well-known/oauth-protected-resource/{mcp_server_name}/mcp"
        print(f"Request: GET {url}")
        print(f"Headers: {json.dumps(forwarded_headers, indent=2)}")

        try:
            response = client.get(url, headers=forwarded_headers)
            print(f"Status: {response.status_code}")

            if response.status_code == 200:
                data = response.json()
                print(f"\nResponse:")
                print(json.dumps(data, indent=2))

                # Verify URLs use external host
                auth_servers = data.get("authorization_servers", [])

                print(f"\n✓ Checking authorization servers use external URL:")
                expected_prefix = f"{external_scheme}://{external_host}"

                for server_url in auth_servers:
                    if server_url.startswith(expected_prefix):
                        print(f"  ✓ {server_url}")
                    else:
                        print(f"  ✗ {server_url} (expected to start with {expected_prefix})")
            else:
                print(f"Error: {response.text}")

        except Exception as e:
            print(f"Error: {e}")

        # Test 3: Client Registration Endpoint
        print("\n\n[Test 3] Client Registration Endpoint")
        print("-" * 70)

        url = f"{litellm_url}/{mcp_server_name}/register"
        print(f"Request: POST {url}")
        print(f"Headers: {json.dumps(forwarded_headers, indent=2)}")

        try:
            response = client.post(
                url,
                headers={**forwarded_headers, "Content-Type": "application/json"},
                json={}
            )
            print(f"Status: {response.status_code}")

            if response.status_code == 200:
                data = response.json()
                print(f"\nResponse:")
                print(json.dumps(data, indent=2))

                # Verify redirect_uris use external host
                redirect_uris = data.get("redirect_uris", [])

                print(f"\n✓ Checking redirect_uris use external URL:")
                expected_prefix = f"{external_scheme}://{external_host}"

                for uri in redirect_uris:
                    if uri.startswith(expected_prefix):
                        print(f"  ✓ {uri}")
                    else:
                        print(f"  ✗ {uri} (expected to start with {expected_prefix})")
            else:
                print(f"Error: {response.text}")

        except Exception as e:
            print(f"Error: {e}")

    print("\n" + "=" * 70)
    print("Testing Complete!")
    print("=" * 70)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Test X-Forwarded-* header support for LiteLLM MCP OAuth"
    )
    parser.add_argument(
        "--litellm-url",
        default="http://localhost:4000",
        help="Internal LiteLLM URL (default: http://localhost:4000)"
    )
    parser.add_argument(
        "--external-host",
        default="proxy.example.com",
        help="External hostname (default: proxy.example.com)"
    )
    parser.add_argument(
        "--external-scheme",
        default="https",
        help="External scheme (default: https)"
    )
    parser.add_argument(
        "--mcp-server",
        default="github",
        help="MCP server name (default: github)"
    )

    args = parser.parse_args()

    test_oauth_endpoints(
        litellm_url=args.litellm_url,
        external_host=args.external_host,
        external_scheme=args.external_scheme,
        mcp_server_name=args.mcp_server,
    )
