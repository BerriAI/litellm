"""
Transform LiteLLM data to FOCUS (FinOps Open Cost & Usage Specification) format.

FOCUS is an open specification for consistent cost and usage datasets.
More info: https://focus.finops.org/
"""

from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import polars as pl

from litellm._logging import verbose_logger
from litellm.types.integrations.focus import FOCUSRecord


class FOCUSTransformer:
    """Transform LiteLLM usage data to FOCUS format."""

    # FOCUS standard values
    CHARGE_CATEGORY_USAGE = "Usage"
    CHARGE_CLASS_STANDARD = "Standard"
    RESOURCE_TYPE_LLM = "LLM"
    SERVICE_CATEGORY = "AI and Machine Learning"
    SERVICE_NAME = "LLM Inference"
    PUBLISHER_NAME = "LiteLLM"
    CONSUMED_UNIT = "Tokens"

    def __init__(
        self,
        include_tags: bool = True,
        include_token_breakdown: bool = True,
    ):
        """
        Initialize FOCUS transformer.
        
        Args:
            include_tags: Whether to include resource tags in output
            include_token_breakdown: Whether to include prompt/completion token breakdown
        """
        self.include_tags = include_tags
        self.include_token_breakdown = include_token_breakdown

    def transform(self, data: pl.DataFrame) -> pl.DataFrame:
        """
        Transform LiteLLM data to FOCUS format.
        
        Filters out records with zero successful_requests.
        
        Args:
            data: Polars DataFrame with LiteLLM usage data
            
        Returns:
            Polars DataFrame with FOCUS formatted records
        """
        if data.is_empty():
            return pl.DataFrame()

        # Filter out records with zero successful_requests
        original_count = len(data)
        if "successful_requests" in data.columns:
            filtered_data = data.filter(pl.col("successful_requests") > 0)
            zero_requests_dropped = original_count - len(filtered_data)
        else:
            filtered_data = data
            zero_requests_dropped = 0

        focus_data = []
        transform_errors = 0
        filtered_count = len(filtered_data)

        for row in filtered_data.iter_rows(named=True):
            try:
                focus_record = self._create_focus_record(row)
                focus_data.append(focus_record)
            except Exception as e:
                transform_errors += 1
                verbose_logger.debug(f"FOCUS Transform: Error processing row: {e}")
                continue

        # Log summary
        if zero_requests_dropped > 0:
            verbose_logger.debug(
                f"FOCUS Transform: Dropped {zero_requests_dropped:,} of {original_count:,} "
                "records with zero successful_requests"
            )

        if transform_errors > 0:
            verbose_logger.warning(
                f"FOCUS Transform: Failed to transform {transform_errors:,} of "
                f"{filtered_count:,} records"
            )

        if len(focus_data) > 0:
            verbose_logger.debug(
                f"FOCUS Transform: Successfully transformed {len(focus_data):,} records"
            )

        return pl.DataFrame(focus_data)

    def _create_focus_record(self, row: dict[str, Any]) -> FOCUSRecord:
        """Create a single FOCUS record from LiteLLM daily spend row."""

        # Parse date for billing period
        usage_date = self._parse_date(row.get("date"))
        
        # Calculate billing period (daily)
        billing_period_start = usage_date.isoformat() if usage_date else None
        billing_period_end = (
            (usage_date + timedelta(days=1)).isoformat() if usage_date else None
        )

        # Calculate total tokens
        prompt_tokens = int(row.get("prompt_tokens", 0) or 0)
        completion_tokens = int(row.get("completion_tokens", 0) or 0)
        total_tokens = prompt_tokens + completion_tokens

        # Get cost
        billed_cost = float(row.get("spend", 0.0) or 0.0)

        # Get provider and model info
        provider = str(row.get("custom_llm_provider", "unknown") or "unknown")
        model = str(row.get("model", "unknown") or "unknown")
        model_group = str(row.get("model_group", "") or "")

        # Get team/account info
        team_id = row.get("team_id")
        team_alias = row.get("team_alias")
        user_id = row.get("user_id")
        api_key = str(row.get("api_key", "") or "")[:8]  # First 8 chars
        api_key_alias = str(row.get("api_key_alias", "") or "")

        # Construct resource ID
        resource_id = self._create_resource_id(
            provider=provider,
            model=model,
            team_id=team_id,
        )

        # Build FOCUS record with required and recommended columns
        focus_record: dict[str, Any] = {
            # Required FOCUS columns
            "BilledCost": billed_cost,
            "BillingPeriodStart": billing_period_start,
            "BillingPeriodEnd": billing_period_end,
            # Charge information
            "ChargeCategory": self.CHARGE_CATEGORY_USAGE,
            "ChargeClass": self.CHARGE_CLASS_STANDARD,
            "ChargeDescription": f"LLM inference using {model} via {provider}",
            "ChargePeriodStart": billing_period_start,
            "ChargePeriodEnd": billing_period_end,
            # Consumption metrics
            "ConsumedQuantity": total_tokens,
            "ConsumedUnit": self.CONSUMED_UNIT,
            # Cost information (same as billed for LLM usage)
            "EffectiveCost": billed_cost,
            "ListCost": billed_cost,
            # Provider information
            "ProviderName": self._normalize_provider_name(provider),
            "PublisherName": self.PUBLISHER_NAME,
            # Resource information
            "ResourceId": resource_id,
            "ResourceName": model,
            "ResourceType": self.RESOURCE_TYPE_LLM,
            # Service information
            "ServiceCategory": self.SERVICE_CATEGORY,
            "ServiceName": self.SERVICE_NAME,
            # Sub-account (team) information
            "SubAccountId": str(team_id) if team_id else None,
            "SubAccountName": str(team_alias) if team_alias else None,
        }

        # Add optional region if available
        region = row.get("region")
        if region:
            focus_record["Region"] = str(region)

        # Add tags if enabled
        if self.include_tags:
            tags = self._build_tags(
                provider=provider,
                model=model,
                model_group=model_group,
                user_id=user_id,
                api_key_prefix=api_key,
                api_key_alias=api_key_alias,
                prompt_tokens=prompt_tokens if self.include_token_breakdown else None,
                completion_tokens=completion_tokens if self.include_token_breakdown else None,
                api_requests=row.get("api_requests"),
                successful_requests=row.get("successful_requests"),
                failed_requests=row.get("failed_requests"),
                cache_creation_tokens=row.get("cache_creation_input_tokens"),
                cache_read_tokens=row.get("cache_read_input_tokens"),
            )
            focus_record["Tags"] = tags

        return FOCUSRecord(focus_record)

    def _parse_date(self, date_str: Any) -> Optional[datetime]:
        """Parse date string from daily spend tables."""
        if date_str is None:
            return None

        if isinstance(date_str, datetime):
            return date_str

        if isinstance(date_str, str):
            try:
                # Parse date string and set to midnight UTC
                return pl.Series([date_str]).str.to_datetime("%Y-%m-%d").item()
            except Exception:
                try:
                    # Fallback: try ISO format parsing
                    return pl.Series([date_str]).str.to_datetime().item()
                except Exception:
                    return None

        return None

    def _create_resource_id(
        self,
        provider: str,
        model: str,
        team_id: Optional[str],
    ) -> str:
        """Create a unique resource ID for FOCUS format."""
        # Format: litellm/{provider}/{team_id or 'default'}/{model}
        safe_provider = self._sanitize_id_component(provider)
        safe_model = self._sanitize_id_component(model)
        safe_team = self._sanitize_id_component(str(team_id) if team_id else "default")
        
        return f"litellm/{safe_provider}/{safe_team}/{safe_model}"

    def _sanitize_id_component(self, component: str) -> str:
        """Sanitize a component for use in resource ID."""
        if not component:
            return "unknown"
        # Replace special characters with hyphens
        import re
        sanitized = re.sub(r"[^a-zA-Z0-9._-]", "-", component.lower())
        # Remove consecutive hyphens
        sanitized = re.sub(r"-+", "-", sanitized)
        return sanitized.strip("-") or "unknown"

    def _normalize_provider_name(self, provider: str) -> str:
        """Normalize provider names to human-readable format."""
        provider_map = {
            "openai": "OpenAI",
            "azure": "Microsoft Azure",
            "azure_ai": "Microsoft Azure",
            "anthropic": "Anthropic",
            "bedrock": "Amazon Web Services",
            "vertex_ai": "Google Cloud",
            "gemini": "Google",
            "cohere": "Cohere",
            "huggingface": "Hugging Face",
            "replicate": "Replicate",
            "together_ai": "Together AI",
            "mistral": "Mistral AI",
            "groq": "Groq",
            "perplexity": "Perplexity",
            "deepseek": "DeepSeek",
            "fireworks_ai": "Fireworks AI",
            "ollama": "Ollama",
        }
        
        normalized = provider.lower().replace("-", "_")
        return provider_map.get(normalized, provider.title())

    def _build_tags(
        self,
        provider: str,
        model: str,
        model_group: Optional[str] = None,
        user_id: Optional[str] = None,
        api_key_prefix: Optional[str] = None,
        api_key_alias: Optional[str] = None,
        prompt_tokens: Optional[int] = None,
        completion_tokens: Optional[int] = None,
        api_requests: Optional[int] = None,
        successful_requests: Optional[int] = None,
        failed_requests: Optional[int] = None,
        cache_creation_tokens: Optional[int] = None,
        cache_read_tokens: Optional[int] = None,
    ) -> dict[str, str]:
        """Build tags dictionary for FOCUS record."""
        tags: dict[str, str] = {
            "litellm:provider": provider,
            "litellm:model": model,
        }
        
        if model_group:
            tags["litellm:model_group"] = model_group
        if user_id:
            tags["litellm:user_id"] = str(user_id)
        if api_key_prefix:
            tags["litellm:api_key_prefix"] = api_key_prefix
        if api_key_alias:
            tags["litellm:api_key_alias"] = api_key_alias
            
        # Token breakdown
        if prompt_tokens is not None:
            tags["litellm:prompt_tokens"] = str(prompt_tokens)
        if completion_tokens is not None:
            tags["litellm:completion_tokens"] = str(completion_tokens)
            
        # Request metrics
        if api_requests is not None:
            tags["litellm:api_requests"] = str(api_requests)
        if successful_requests is not None:
            tags["litellm:successful_requests"] = str(successful_requests)
        if failed_requests is not None:
            tags["litellm:failed_requests"] = str(failed_requests)
            
        # Cache metrics
        if cache_creation_tokens:
            tags["litellm:cache_creation_tokens"] = str(cache_creation_tokens)
        if cache_read_tokens:
            tags["litellm:cache_read_tokens"] = str(cache_read_tokens)

        return tags
