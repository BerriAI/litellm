"""
Cost tracking for JSON-configured providers.

Calculates costs based on usage and provider-specific pricing.
"""

from typing import Any, Optional

from litellm._logging import verbose_logger


class CostTracker:
    """
    Track costs for JSON-configured providers.
    
    Supports:
    - Per-image pricing (image generation)
    - Per-token pricing (chat/completions)
    - Per-request pricing
    """

    @staticmethod
    def calculate_image_generation_cost(
        response: Any, model: str, cost_config: Any
    ) -> float:
        """
        Calculate cost for image generation.
        
        Args:
            response: LiteLLM ImageResponse
            model: Model name used
            cost_config: CostTrackingConfig object
        
        Returns:
            Total cost in USD
        """
        if not cost_config.enabled:
            verbose_logger.debug("Cost tracking disabled for this provider")
            return 0.0

        try:
            # Get number of images generated
            num_images = len(response.data) if hasattr(response, "data") and response.data else 0

            if num_images == 0:
                verbose_logger.debug("No images in response, cost is 0")
                return 0.0

            # Get cost per image for this model
            cost_per_image = cost_config.cost_per_image.get(model, 0.0)

            if cost_per_image == 0.0:
                verbose_logger.warning(
                    f"No cost_per_image configured for model '{model}', defaulting to 0"
                )

            # Calculate total cost
            total_cost = num_images * cost_per_image

            verbose_logger.debug(
                f"Cost calculation: {num_images} images × ${cost_per_image} = ${total_cost}"
            )

            return total_cost

        except Exception as e:
            verbose_logger.error(f"Error calculating image generation cost: {e}")
            return 0.0

    @staticmethod
    def calculate_completion_cost(
        response: Any, model: str, cost_config: Any
    ) -> float:
        """
        Calculate cost for completion endpoints (chat, completions).
        
        Args:
            response: LiteLLM completion response
            model: Model name used
            cost_config: CostTrackingConfig object
        
        Returns:
            Total cost in USD
        """
        if not cost_config.enabled:
            return 0.0

        try:
            # Extract token usage
            usage = getattr(response, "usage", None)
            if not usage:
                verbose_logger.debug("No usage information in response")
                return 0.0

            prompt_tokens = getattr(usage, "prompt_tokens", 0)
            completion_tokens = getattr(usage, "completion_tokens", 0)

            # Get costs per token for this model
            if not cost_config.cost_per_token:
                verbose_logger.warning("No cost_per_token configuration found")
                return 0.0

            token_costs = cost_config.cost_per_token.get(model, {})
            if not token_costs:
                verbose_logger.warning(f"No token costs configured for model '{model}'")
                return 0.0

            prompt_cost_per_token = token_costs.get("prompt", 0.0)
            completion_cost_per_token = token_costs.get("completion", 0.0)

            # Calculate total cost
            total_cost = (prompt_tokens * prompt_cost_per_token) + (
                completion_tokens * completion_cost_per_token
            )

            verbose_logger.debug(
                f"Cost calculation: ({prompt_tokens} prompt tokens × ${prompt_cost_per_token}) + "
                f"({completion_tokens} completion tokens × ${completion_cost_per_token}) = ${total_cost}"
            )

            return total_cost

        except Exception as e:
            verbose_logger.error(f"Error calculating completion cost: {e}")
            return 0.0

    @staticmethod
    def add_cost_to_response(response: Any, cost: float) -> Any:
        """
        Add cost information to response object.
        
        Args:
            response: LiteLLM response object
            cost: Calculated cost
        
        Returns:
            Response with cost added to _hidden_params
        """
        try:
            if not hasattr(response, "_hidden_params"):
                response._hidden_params = {}
            response._hidden_params["response_cost"] = cost
            
            verbose_logger.debug(f"Added cost ${cost} to response")
        except Exception as e:
            verbose_logger.error(f"Error adding cost to response: {e}")
        
        return response
