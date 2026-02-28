"""
Model inference for self-hosted providers.

When using self-hosted providers (hosted_vllm, openai_like, ollama, etc.) with custom model names,
this module infers model capabilities from similar models already in the registry.

Example:
    hosted_vllm/my-custom-llama-70b  → infers from known llama-70b variants
    openai_like/mistral-7b-custom    → infers from known mistral-7b variants
"""
from typing import Dict, List, Optional, Any
import re
from litellm._logging import verbose_logger
import litellm


# Self-hosted providers that support model inference
SELF_HOSTED_PROVIDERS = [
    "hosted_vllm",
    "openai_like",
    "ollama",
    "ollama_chat",
    "lm_studio",
    "llamafile",
]


def extract_base_model_patterns(model_name: str) -> List[str]:
    """
    Extract potential base model patterns from a model name.
    
    Args:
        model_name: Original model name (e.g., "my-custom-llama-3.1-70b-instruct")
    
    Returns:
        List of patterns to search for, from most specific to least specific
        
    Example:
        "my-custom-llama-3.1-70b-instruct" → [
            "llama-3.1-70b-instruct",
            "llama-3.1-70b",
            "llama-3.1",
            "llama-3",
            "llama"
        ]
    """
    # Clean up the model name
    name = model_name.lower().strip()
    
    # Remove common prefixes
    for prefix in ["my-", "custom-", "fine-tuned-", "ft-"]:
        if name.startswith(prefix):
            name = name[len(prefix):]
    
    patterns = []
    
    # Try to extract model family patterns
    # Match patterns like: llama-3.1-70b, mistral-7b, qwen-72b, etc.
    
    # First, find size if it exists (e.g., 7b, 70b, 72b)
    size_match = re.search(r'(\d+[bkm])\b', name, re.IGNORECASE)
    size = size_match.group(1).lower() if size_match else None
    
    # Find model family
    family_match = re.search(r'\b(llama|mistral|qwen|yi|phi|gemma|mixtral)', name, re.IGNORECASE)
    family = family_match.group(1).lower() if family_match else None
    
    # Find version (numbers with optional dots, not followed by b/k/m)
    version_match = re.search(r'[-_](\d+(?:\.\d+)?)(?!b|k|m)', name, re.IGNORECASE)
    version = version_match.group(1) if version_match else None
    
    # Find variant (instruct, chat, base, etc.)
    variant_match = re.search(r'[-_](instruct|chat|base|code)', name, re.IGNORECASE)
    variant = variant_match.group(1).lower() if variant_match else None
    
    if family:
        
        # Build patterns from most specific to least specific
        if variant and size and version:
            patterns.append(f"{family}-{version}-{size}-{variant}")
        if size and version:
            patterns.append(f"{family}-{version}-{size}")
        if version and size:
            patterns.append(f"{family}-{size}")  # Also try without version
        if version:
            patterns.append(f"{family}-{version}")
        if size:
            patterns.append(f"{family}-{size}")
        patterns.append(family)
    else:
        # Fallback: just use the name with common suffixes removed
        clean_name = re.sub(r'[-_](instruct|chat|base|v\d+|fp16|gguf).*$', '', name, flags=re.IGNORECASE)
        if clean_name and clean_name != name:
            patterns.append(clean_name)
        patterns.append(name)
    
    verbose_logger.debug(f"Extracted patterns from '{model_name}': {patterns}")
    return patterns


def find_similar_models(patterns: List[str]) -> List[Dict[str, Any]]:
    """
    Search model_cost for models matching the given patterns.
    
    Args:
        patterns: List of patterns to search for
    
    Returns:
        List of matching model info dicts from model_cost
    """
    matches = []
    seen_keys = set()
    
    for pattern in patterns:
        pattern_lower = pattern.lower()
        
        for model_key, model_info in litellm.model_cost.items():
            if model_key in seen_keys:
                continue
                
            model_key_lower = model_key.lower()
            
            # Check if pattern is in the model key
            if pattern_lower in model_key_lower:
                # Skip if this is an inferred or not_found marker
                if model_info.get("_inferred") or model_info.get("_not_found"):
                    continue
                
                matches.append({
                    "key": model_key,
                    "info": model_info,
                    "pattern": pattern
                })
                seen_keys.add(model_key)
        
        # If we found matches with this pattern, stop searching
        # (we want the most specific matches)
        if matches:
            break
    
    verbose_logger.debug(f"Found {len(matches)} similar models")
    return matches


def aggregate_capabilities(matches: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Aggregate capabilities from multiple matching models.
    
    Uses MAX for numeric values and ANY for boolean capabilities.
    
    Args:
        matches: List of matching model dicts
    
    Returns:
        Aggregated model info dict
    """
    if not matches:
        return {}
    
    aggregated: Dict[str, Any] = {
        "input_cost_per_token": 0.0,  # Zero out costs for self-hosted
        "output_cost_per_token": 0.0,
    }
    
    # Numeric fields - use MAX
    numeric_fields = [
        "max_tokens",
        "max_input_tokens",
        "max_output_tokens",
        "output_vector_size",
    ]
    
    # Boolean fields - use ANY (True if any model supports it)
    boolean_fields = [
        "supports_system_messages",
        "supports_response_schema",
        "supports_vision",
        "supports_function_calling",
        "supports_tool_choice",
        "supports_assistant_prefill",
        "supports_prompt_caching",
        "supports_audio_input",
        "supports_audio_output",
        "supports_pdf_input",
        "supports_web_search",
        "supports_url_context",
        "supports_reasoning",
        "supports_computer_use",
    ]
    
    # Aggregate numeric fields
    for field in numeric_fields:
        values = [m["info"].get(field) for m in matches if m["info"].get(field) is not None]
        if values:
            aggregated[field] = max(values)
    
    # Aggregate boolean fields
    for field in boolean_fields:
        values = [m["info"].get(field) for m in matches if m["info"].get(field) is not None]
        if values:
            aggregated[field] = any(values)
    
    # Use mode from first match
    if matches[0]["info"].get("mode"):
        aggregated["mode"] = matches[0]["info"]["mode"]
    
    verbose_logger.debug(f"Aggregated capabilities from {len(matches)} models")
    return aggregated


def infer_model_capabilities(
    model: str,
    custom_llm_provider: str
) -> Optional[Dict[str, Any]]:
    """
    Infer model capabilities from similar models in the registry.
    
    This is the main entry point for model inference.
    
    Args:
        model: Model name (e.g., "my-custom-llama-70b")
        custom_llm_provider: Provider name (e.g., "hosted_vllm")
    
    Returns:
        Inferred model info dict or None if inference fails
    """
    # Check if provider supports inference
    if custom_llm_provider not in SELF_HOSTED_PROVIDERS:
        return None
    
    # Check cache first
    cache_key = f"{custom_llm_provider}/{model}"
    if cache_key in litellm.model_cost:
        cached = litellm.model_cost[cache_key]
        if cached.get("_not_found"):
            verbose_logger.debug(f"Model {cache_key} previously marked as not found")
            return None
        return cached
    
    verbose_logger.info(f"Attempting to infer capabilities for {cache_key}")
    
    # Extract patterns to search for
    patterns = extract_base_model_patterns(model)
    
    # Find similar models
    matches = find_similar_models(patterns)
    
    if not matches:
        verbose_logger.info(f"No similar models found for {model}, caching not_found marker")
        # Cache negative result
        litellm.model_cost[cache_key] = {"_not_found": True}
        return None
    
    # Aggregate capabilities
    inferred_info = aggregate_capabilities(matches)
    
    if not inferred_info:
        litellm.model_cost[cache_key] = {"_not_found": True}
        return None
    
    # Add metadata
    inferred_info["_inferred"] = True
    inferred_info["_inferred_from"] = [m["key"] for m in matches[:3]]  # Track sources
    inferred_info["litellm_provider"] = custom_llm_provider
    inferred_info["key"] = cache_key
    
    # Cache result
    litellm.model_cost[cache_key] = inferred_info
    
    verbose_logger.info(
        f"Successfully inferred capabilities for {cache_key} from {len(matches)} similar models"
    )
    
    return inferred_info
