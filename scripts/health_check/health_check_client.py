#!/usr/bin/env python3
"""
LiteLLM Health Check Client

A sentinel health check tool that tests all configured models on a LiteLLM proxy.
This script:
- Can read models from YAML config file or fetch from proxy API
- Sends a simple test request to each model concurrently
- Reports health status for each model
- Supports both chat/completion and embedding models
"""

import asyncio
import json
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import httpx
import yaml

# Default prompt for health checks - exactly 100k characters
# Generate a repeating pattern to reach exactly 100,000 characters
_base_text = "This is a health check test prompt for LiteLLM proxy. "
_repeat_count = (100000 // len(_base_text)) + 1
_DEFAULT_COMPLETION_PROMPT = (_base_text * _repeat_count)[:100000]

# Default embedding text - also exactly 100k characters
_embedding_base_text = "This is a test for vectorization. "
_embedding_repeat_count = (100000 // len(_embedding_base_text)) + 1
_DEFAULT_EMBEDDING_TEXT = (_embedding_base_text * _embedding_repeat_count)[:100000]


class LiteLLMHealthCheckClient:
    """Client for health checking LiteLLM proxy models."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        timeout: int = 120,  # Match Go implementation's 120s timeout
        completion_prompt: str = _DEFAULT_COMPLETION_PROMPT,  # Default ~100k chars
        embedding_text: str = _DEFAULT_EMBEDDING_TEXT,  # Default ~100k chars
        custom_auth_header: Optional[str] = None,
    ):
        """
        Initialize the health check client.

        Args:
            base_url: Base URL of the LiteLLM proxy (e.g., https://litellm.example.com)
            api_key: API key for authentication
            timeout: Request timeout in seconds (default: 120, matching Go implementation)
            completion_prompt: Test prompt for chat/completion models
            embedding_text: Test text for embedding models
            custom_auth_header: Optional custom header name for authentication (e.g., "x-ifood-requester-service").
                If provided, uses this header instead of standard "Authorization" header.
        """
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.completion_prompt = completion_prompt
        self.embedding_text = embedding_text
        
        # Debug: Print prompt/text lengths
        print(f"DEBUG: Completion prompt length: {len(self.completion_prompt)} characters", file=sys.stderr)
        print(f"DEBUG: Embedding text length: {len(self.embedding_text)} characters", file=sys.stderr)
        
        # Support custom auth header for proxies with custom authentication
        # Handle both None and empty string
        if custom_auth_header and custom_auth_header.strip():
            custom_auth_header = custom_auth_header.strip()
            self.headers = {
                custom_auth_header: f"Bearer {api_key}",
                "Content-Type": "application/json",
            }
            print(f"Using custom auth header: {custom_auth_header}", file=sys.stderr)
        else:
            self.headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }
            print("Using standard Authorization header", file=sys.stderr)

    def load_models_from_yaml(self, yaml_path: str) -> List[Dict]:
        """
        Load models from a YAML config file (similar to Go implementation).

        Args:
            yaml_path: Path to the YAML config file

        Returns:
            List of model dictionaries with 'id' and 'mode' keys
        """
        try:
            with open(yaml_path, "r") as f:
                config = yaml.safe_load(f)

            model_list = config.get("model_list", [])
            models = []

            for entry in model_list:
                model_name = entry.get("model_name", "")
                litellm_params = entry.get("litellm_params", {})
                model_info = litellm_params.get("model_info", {})
                mode = model_info.get("mode", "")

                # Use model_name as the ID (this is what gets sent to the API)
                models.append(
                    {
                        "id": model_name,
                        "mode": mode.lower() if mode else "",
                        "provider": model_info.get("provider", ""),
                    }
                )

            return models
        except Exception as e:
            print(f"Error loading models from YAML file {yaml_path}: {e}", file=sys.stderr)
            return []

    async def fetch_models(self, client: httpx.AsyncClient) -> List[Dict]:
        """
        Fetch all available models from the proxy API.

        Returns:
            List of model dictionaries with 'id' and 'mode' keys
        """
        try:
            # Try /v1/models first (OpenAI-compatible endpoint)
            response = await client.get(
                f"{self.base_url}/v1/models",
                headers=self.headers,
                timeout=self.timeout,
            )
            response.raise_for_status()
            data = response.json()
            models_data = data.get("data", [])
            models = []
            for m in models_data:
                models.append({"id": m["id"], "mode": "", "provider": ""})
            return models
        except Exception as e:
            print(f"Error fetching models from /v1/models: {e}", file=sys.stderr)
            # Fallback to /model/info endpoint which has more details
            try:
                response = await client.get(
                    f"{self.base_url}/model/info",
                    headers=self.headers,
                    timeout=self.timeout,
                )
                response.raise_for_status()
                data = response.json()
                if isinstance(data, dict) and "data" in data:
                    models_data = data["data"]
                elif isinstance(data, list):
                    models_data = data
                else:
                    models_data = []

                models = []
                for m in models_data:
                    model_info = m.get("model_info", {})
                    mode = model_info.get("mode", "")
                    models.append(
                        {
                            "id": m.get("model_name", m.get("id", "unknown")),
                            "mode": mode.lower() if mode else "",
                            "provider": model_info.get("provider", ""),
                        }
                    )
                return models
            except Exception as e2:
                print(f"Error fetching models from /model/info: {e2}", file=sys.stderr)
                return []

    async def check_model_health(
        self, client: httpx.AsyncClient, model: Dict
    ) -> Tuple[str, Dict]:
        """
        Check health of a single model by sending a test request.

        Args:
            client: HTTP client
            model: Model dictionary with 'id' and 'mode' keys

        Returns:
            Tuple of (model_id, result_dict)
        """
        model_id = model["id"]
        mode = model.get("mode", "")

        start_time = time.time()
        result = {
            "model": model_id,
            "healthy": False,
            "error": None,
            "response_time_ms": None,
            "mode": mode,
        }

        try:
            # Determine if this is an embedding model
            # Check mode first (from config), then fall back to name-based detection
            is_embedding = (
                mode == "embedding"
                or any(
                    keyword in model_id.lower()
                    for keyword in ["embedding", "embed", "text-embedding"]
                )
            )

            if is_embedding:
                # Test embedding endpoint (matching Go implementation)
                embedding_text_length = len(self.embedding_text)
                print(f"DEBUG: Sending embedding text of length {embedding_text_length} chars to model {model_id}", file=sys.stderr)
                embedding_response = await client.post(
                    f"{self.base_url}/v1/embeddings",
                    headers=self.headers,
                    json={
                        "model": model_id,
                        "input": self.embedding_text,
                    },
                    timeout=self.timeout,
                )
                embedding_response.raise_for_status()
                embedding_data = embedding_response.json()
                dimensions = 0
                if "data" in embedding_data and len(embedding_data["data"]) > 0:
                    dimensions = len(embedding_data["data"][0].get("embedding", []))

                result["healthy"] = True
                result["mode"] = "embedding"
                result["dimensions"] = dimensions
            else:
                # Test chat completion endpoint (matching Go implementation)
                prompt_length = len(self.completion_prompt)
                print(f"DEBUG: Sending prompt of length {prompt_length} chars to model {model_id}", file=sys.stderr)
                completion_response = await client.post(
                    f"{self.base_url}/v1/chat/completions",
                    headers=self.headers,
                    json={
                        "model": model_id,
                        "messages": [
                            {"role": "user", "content": self.completion_prompt}
                        ],
                        "max_tokens": 10,  # Minimal tokens for health check
                    },
                    timeout=self.timeout,
                )
                completion_response.raise_for_status()
                completion_data = completion_response.json()
                response_text = ""
                if "choices" in completion_data and len(completion_data["choices"]) > 0:
                    response_text = (
                        completion_data["choices"][0]
                        .get("message", {})
                        .get("content", "")
                    )

                result["healthy"] = True
                result["mode"] = "chat"
                result["response_text"] = response_text[:100]  # Truncate for display

            elapsed_ms = (time.time() - start_time) * 1000
            result["response_time_ms"] = round(elapsed_ms, 2)

        except httpx.HTTPStatusError as e:
            result["error"] = f"HTTP {e.response.status_code}: {e.response.text[:200]}"
        except httpx.TimeoutException:
            result["error"] = f"Request timeout after {self.timeout}s"
        except Exception as e:
            result["error"] = str(e)[:200]

        return model_id, result

    async def run_health_checks(
        self,
        models: Optional[List[Dict]] = None,
        models_only: Optional[List[str]] = None,
    ) -> Dict[str, Dict]:
        """
        Run health checks on all models concurrently.

        Args:
            models: Optional list of models to check. If None, fetches from proxy.
            models_only: Optional list of model IDs to check. If set, only these
                models are health-checked (must exist in the models list).

        Returns:
            Dictionary mapping model_id to health check result
        """
        async with httpx.AsyncClient() as client:
            if models is None:
                models = await self.fetch_models(client)

            if not models:
                print("No models found to health check", file=sys.stderr)
                return {}

            if models_only:
                allowlist = {m.strip() for m in models_only if m and m.strip()}
                models = [m for m in models if m.get("id") in allowlist]
                print(
                    f"Filtering to only check {len(models)} models: {', '.join(sorted(allowlist))}",
                    file=sys.stderr,
                )
                if not models:
                    print(
                        "No models matched LITELLM_MODELS_ONLY filter",
                        file=sys.stderr,
                    )
                    return {}

            print(f"Running health checks on {len(models)} models...", file=sys.stderr)

            # Run all health checks concurrently
            tasks = [self.check_model_health(client, model) for model in models]
            results_list = await asyncio.gather(*tasks, return_exceptions=True)

            # Convert to dictionary format
            results = {}
            for result in results_list:
                if isinstance(result, Exception):
                    print(
                        f"Exception in health check task: {result}", file=sys.stderr
                    )
                    continue
                # Type narrowing: after checking it's not an Exception, it's a Tuple
                if isinstance(result, tuple) and len(result) == 2:
                    model_id, result_dict = result
                    results[model_id] = result_dict

            return results

    def print_results(self, results: Dict[str, Dict], json_output: bool = False):
        """
        Print health check results.

        Args:
            results: Dictionary of health check results
            json_output: If True, output as JSON
        """
        if json_output:
            print(json.dumps(results, indent=2))
            return

        healthy_count = sum(1 for r in results.values() if r.get("healthy"))
        unhealthy_count = len(results) - healthy_count

        # Print detailed results for each model (matching Go output format)
        print(f"\n{'='*60}", file=sys.stderr)
        print(f"Starting health check queries\n", file=sys.stderr)

        for model_id, result in results.items():
            if result.get("healthy"):
                if result.get("mode") == "embedding":
                    dimensions = result.get("dimensions", 0)
                    print(
                        f"---- {model_id} ----\n✅ Success. "
                        f"Generated embedding vector with {dimensions} dimensions.\n\n",
                        file=sys.stderr,
                    )
                else:
                    response_text = result.get("response_text", "")
                    print(
                        f"---- {model_id} ----\n✅ Success. "
                        f"Response:\n{response_text}\n\n",
                        file=sys.stderr,
                    )
            else:
                error = result.get("error", "Unknown error")
                print(f"---- {model_id} ----\n❌ ERROR: {error}\n\n", file=sys.stderr)

        print(f"{'='*60}", file=sys.stderr)
        print(f"Health Check Summary", file=sys.stderr)
        print(f"{'='*60}", file=sys.stderr)
        print(f"Total models: {len(results)}", file=sys.stderr)
        print(f"Healthy: {healthy_count}", file=sys.stderr)
        print(f"Unhealthy: {unhealthy_count}", file=sys.stderr)
        print(f"{'='*60}\n", file=sys.stderr)

        # Exit with non-zero code if any models are unhealthy
        if unhealthy_count > 0:
            sys.exit(1)
        else:
            sys.exit(0)


async def main():
    """Main entry point."""
    base_url = os.environ.get("LITELLM_BASE_URL", "http://localhost:4000")
    api_key = os.environ.get("LITELLM_API_KEY", "sk-1234")
    yaml_path = os.environ.get("LITELLM_MODELS_YAML")
    custom_auth_header = os.environ.get("LITELLM_CUSTOM_AUTH_HEADER")  # e.g., "x-ifood-requester-service"
    
    # Debug: Print custom auth header value if set
    if custom_auth_header:
        print(f"Custom auth header from env: '{custom_auth_header}'", file=sys.stderr)

    if not base_url:
        print("Error: LITELLM_BASE_URL environment variable not set", file=sys.stderr)
        sys.exit(1)

    if not api_key:
        print("Error: LITELLM_API_KEY environment variable not set", file=sys.stderr)
        sys.exit(1)

    timeout = int(os.environ.get("LITELLM_TIMEOUT", "120"))  # Match Go's 120s default
    completion_prompt = os.environ.get(
        "LITELLM_COMPLETION_PROMPT", _DEFAULT_COMPLETION_PROMPT
    )
    embedding_text = os.environ.get(
        "LITELLM_EMBEDDING_TEXT", _DEFAULT_EMBEDDING_TEXT
    )
    json_output = os.environ.get("LITELLM_JSON_OUTPUT", "").lower() == "true"
    # Optional: only health-check these model IDs (comma-separated). E.g.:
    # LITELLM_MODELS_ONLY=claude-3.7-sonnet,claude-3.5-sonnet,claude-4.5-haiku
    models_only_raw = os.environ.get("LITELLM_MODELS_ONLY", "")
    models_only = [m.strip() for m in models_only_raw.split(",") if m.strip()] or None

    client = LiteLLMHealthCheckClient(
        base_url=base_url,
        api_key=api_key,
        timeout=timeout,
        completion_prompt=completion_prompt,
        embedding_text=embedding_text,
        custom_auth_header=custom_auth_header,
    )

    # Load models from YAML if provided, otherwise fetch from API
    models = None
    if yaml_path:
        models = client.load_models_from_yaml(yaml_path)
        if models:
            print(
                f"Successfully loaded {len(models)} models from {yaml_path}",
                file=sys.stderr,
            )

    results = await client.run_health_checks(models=models, models_only=models_only)
    client.print_results(results, json_output=json_output)


if __name__ == "__main__":
    asyncio.run(main())
