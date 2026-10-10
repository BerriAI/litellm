import enum


class CommonProxyErrors(str, enum.Enum):
    db_not_connected_error = (
        "DB not connected. This endpoint needs a database; set DATABASE_URL to a "
        "PostgreSQL connection string (postgresql://...) to enable it. "
        "See https://docs.litellm.ai/docs/proxy/virtual_keys"
    )
    no_llm_router = "No models configured on proxy"
    not_allowed_access = "Admin-only endpoint. Not allowed to access this."
    not_premium_user = "You must be a LiteLLM Enterprise user to use this feature. If you have a license please set `LITELLM_LICENSE` in your env. Get a 7 day trial key here: https://www.litellm.ai/enterprise#trial. \nPricing: https://www.litellm.ai/#pricing"
    max_parallel_request_limit_reached = "Crossed TPM / RPM / Max Parallel Request Limit"
    missing_enterprise_package = "Missing litellm-enterprise package. Please install it to use this feature. Run `pip install litellm-enterprise`"
    missing_enterprise_package_docker = "This uses the enterprise folder - only available on the Docker image."
