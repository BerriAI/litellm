`litellm-router` scaffolds the model-list setup and deployment lookup portion of Python's `litellm.Router`. `Router::from_model_list(&config.model_list)` maps configured public names to provider deployments. Programmatic callers can collect `(String, Deployment)` entries into a `Router`

Lookup is exact and returns `None` for an unknown name. This extraction preserves the gateway's existing behavior: the last entry wins when public names repeat. Multiple deployments per model group, routing strategies, retries, cooldowns, and fallbacks are not implemented yet

The router owns deployment configuration and selection. The gateway handles HTTP errors and responses, while `core` executes provider calls and resolves credentials
