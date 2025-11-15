from __future__ import annotations

from typing import Any, Dict, List

from litellm.types.proxy.public_endpoints.public_endpoints import (
    ProviderCreateInfo,
    ProviderCredentialField,
)
from litellm.types.utils import LlmProviders

DEFAULT_MODEL_PLACEHOLDER = "gpt-3.5-turbo"

_FALLBACK_FIELDS: List[Dict[str, Any]] = [
    {
        "key": "api_base",
        "label": "API Base",
        "field_type": "text",
        "required": False,
    },
    {
        "key": "api_key",
        "label": "API Key",
        "field_type": "password",
        "required": False,
    },
]

PROVIDER_BASE_INFO: Dict[str, Dict[str, Any]] = {
    "AIML": {
        "provider_display_name": "AI/ML API",
        "litellm_provider": "aiml",
        "default_model_placeholder": "aiml/flux-pro/v1.1",
    },
    "Anthropic": {
        "provider_display_name": "Anthropic",
        "litellm_provider": "anthropic",
        "default_model_placeholder": "claude-3-opus",
    },
    "AssemblyAI": {
        "provider_display_name": "AssemblyAI",
        "litellm_provider": "assemblyai",
    },
    "Azure": {
        "provider_display_name": "Azure",
        "litellm_provider": "azure",
        "default_model_placeholder": "azure/my-deployment",
    },
    "Azure_AI_Studio": {
        "provider_display_name": "Azure AI Foundry (Studio)",
        "litellm_provider": "azure_ai",
        "default_model_placeholder": "azure_ai/command-r-plus",
    },
    "Bedrock": {
        "provider_display_name": "Amazon Bedrock",
        "litellm_provider": "bedrock",
        "default_model_placeholder": "claude-3-opus",
    },
    "Cerebras": {
        "provider_display_name": "Cerebras",
        "litellm_provider": "cerebras",
    },
    "Cohere": {
        "provider_display_name": "Cohere",
        "litellm_provider": "cohere",
    },
    "Dashscope": {
        "provider_display_name": "Dashscope",
        "litellm_provider": "dashscope",
    },
    "Databricks": {
        "provider_display_name": "Databricks (Qwen API)",
        "litellm_provider": "databricks",
    },
    "DeepInfra": {
        "provider_display_name": "DeepInfra",
        "litellm_provider": "deepinfra",
        "default_model_placeholder": "deepinfra/<any-model-on-deepinfra>",
    },
    "Deepgram": {
        "provider_display_name": "Deepgram",
        "litellm_provider": "deepgram",
    },
    "Deepseek": {
        "provider_display_name": "Deepseek",
        "litellm_provider": "deepseek",
    },
    "ElevenLabs": {
        "provider_display_name": "ElevenLabs",
        "litellm_provider": "elevenlabs",
    },
    "FalAI": {
        "provider_display_name": "Fal AI",
        "litellm_provider": "fal_ai",
        "default_model_placeholder": "fal_ai/fal-ai/flux-pro/v1.1-ultra",
    },
    "FireworksAI": {
        "provider_display_name": "Fireworks AI",
        "litellm_provider": "fireworks_ai",
    },
    "Google_AI_Studio": {
        "provider_display_name": "Google AI Studio",
        "litellm_provider": "gemini",
        "default_model_placeholder": "gemini-pro",
    },
    "GradientAI": {
        "provider_display_name": "GradientAI",
        "litellm_provider": "gradient_ai",
    },
    "Groq": {
        "provider_display_name": "Groq",
        "litellm_provider": "groq",
    },
    "Hosted_Vllm": {
        "provider_display_name": "vllm",
        "litellm_provider": "hosted_vllm",
    },
    "Infinity": {
        "provider_display_name": "Infinity",
        "litellm_provider": "infinity",
    },
    "JinaAI": {
        "provider_display_name": "Jina AI",
        "litellm_provider": "jina_ai",
        "default_model_placeholder": "jina_ai/",
    },
    "MistralAI": {
        "provider_display_name": "Mistral AI",
        "litellm_provider": "mistral",
    },
    "Ollama": {
        "provider_display_name": "Ollama",
        "litellm_provider": "ollama",
    },
    "OpenAI": {
        "provider_display_name": "OpenAI",
        "litellm_provider": "openai",
    },
    "OpenAI_Compatible": {
        "provider_display_name": "OpenAI-Compatible Endpoints (Together AI, etc.)",
        "litellm_provider": "openai",
    },
    "OpenAI_Text": {
        "provider_display_name": "OpenAI Text Completion",
        "litellm_provider": "text-completion-openai",
    },
    "OpenAI_Text_Compatible": {
        "provider_display_name": "OpenAI-Compatible Text Completion Models (Together AI, etc.)",
        "litellm_provider": "text-completion-openai",
    },
    "Openrouter": {
        "provider_display_name": "Openrouter",
        "litellm_provider": "openrouter",
    },
    "Oracle": {
        "provider_display_name": "Oracle Cloud Infrastructure (OCI)",
        "litellm_provider": "oci",
        "default_model_placeholder": "oci/xai.grok-4",
    },
    "Perplexity": {
        "provider_display_name": "Perplexity",
        "litellm_provider": "perplexity",
    },
    "SageMaker": {
        "provider_display_name": "AWS SageMaker",
        "litellm_provider": "sagemaker_chat",
        "default_model_placeholder": "sagemaker/jumpstart-dft-meta-textgeneration-llama-2-7b",
    },
    "Sambanova": {
        "provider_display_name": "Sambanova",
        "litellm_provider": "sambanova",
    },
    "Snowflake": {
        "provider_display_name": "Snowflake",
        "litellm_provider": "snowflake",
        "default_model_placeholder": "snowflake/mistral-7b",
    },
    "TogetherAI": {
        "provider_display_name": "TogetherAI",
        "litellm_provider": "together_ai",
    },
    "Triton": {
        "provider_display_name": "Triton",
        "litellm_provider": "triton",
    },
    "Vertex_AI": {
        "provider_display_name": "Vertex AI (Anthropic, Gemini, etc.)",
        "litellm_provider": "vertex_ai",
        "default_model_placeholder": "gemini-pro",
    },
    "VolcEngine": {
        "provider_display_name": "VolcEngine",
        "litellm_provider": "volcengine",
        "default_model_placeholder": "volcengine/<any-model-on-volcengine>",
    },
    "Voyage": {
        "provider_display_name": "Voyage AI",
        "litellm_provider": "voyage",
        "default_model_placeholder": "voyage/",
    },
    "xAI": {
        "provider_display_name": "xAI",
        "litellm_provider": "xai",
    },
}

PROVIDER_CREDENTIAL_FIELDS: Dict[str, List[Dict[str, Any]]] = {
    "OpenAI": [
        {
            "key": "api_base",
            "label": "API Base",
            "field_type": "text",
            "placeholder": "https://api.openai.com/v1",
            "tooltip": "Common endpoints: https://api.openai.com/v1, https://eu.api.openai.com, https://us.api.openai.com",
            "default_value": "https://api.openai.com/v1",
        },
        {
            "key": "organization",
            "label": "OpenAI Organization ID",
            "placeholder": "[OPTIONAL] my-unique-org",
        },
        {
            "key": "api_key",
            "label": "OpenAI API Key",
            "field_type": "password",
            "required": True,
        },
    ],
    "OpenAI_Text": [
        {
            "key": "api_base",
            "label": "API Base",
            "field_type": "text",
            "placeholder": "https://api.openai.com/v1",
            "tooltip": "Common endpoints: https://api.openai.com/v1, https://eu.api.openai.com, https://us.api.openai.com",
            "default_value": "https://api.openai.com/v1",
        },
        {
            "key": "organization",
            "label": "OpenAI Organization ID",
            "placeholder": "[OPTIONAL] my-unique-org",
        },
        {
            "key": "api_key",
            "label": "OpenAI API Key",
            "field_type": "password",
            "required": True,
        },
    ],
    "Vertex_AI": [
        {
            "key": "vertex_project",
            "label": "Vertex Project",
            "placeholder": "adroit-cadet-1234..",
            "required": True,
        },
        {
            "key": "vertex_location",
            "label": "Vertex Location",
            "placeholder": "us-east-1",
            "required": True,
        },
        {
            "key": "vertex_credentials",
            "label": "Vertex Credentials",
            "field_type": "upload",
            "required": True,
        },
    ],
    "AssemblyAI": [
        {
            "key": "api_base",
            "label": "API Base",
            "field_type": "select",
            "required": True,
            "options": [
                "https://api.assemblyai.com",
                "https://api.eu.assemblyai.com",
            ],
        },
        {
            "key": "api_key",
            "label": "AssemblyAI API Key",
            "field_type": "password",
            "required": True,
        },
    ],
    "Azure": [
        {
            "key": "api_base",
            "label": "API Base",
            "placeholder": "https://...",
            "required": True,
        },
        {
            "key": "api_version",
            "label": "API Version",
            "placeholder": "2023-07-01-preview",
            "tooltip": "By default litellm will use the latest version. If you want to use a different version, you can specify it here",
        },
        {
            "key": "base_model",
            "label": "Base Model",
            "placeholder": "azure/gpt-3.5-turbo",
        },
        {
            "key": "api_key",
            "label": "Azure API Key",
            "field_type": "password",
            "placeholder": "Enter your Azure API Key",
        },
        {
            "key": "azure_ad_token",
            "label": "Azure AD Token",
            "field_type": "password",
            "placeholder": "Enter your Azure AD Token",
        },
    ],
    "Azure_AI_Studio": [
        {
            "key": "api_base",
            "label": "API Base",
            "placeholder": "https://<test>.openai.azure.com/openai/deployments/gpt-4o/chat/completions?api-version=2024-10-21",
            "tooltip": "Enter your full Target URI from Azure Foundry here. Example:  https://litellm8397336933.openai.azure.com/openai/deployments/gpt-4o/chat/completions?api-version=2024-10-21",
            "required": True,
        },
        {
            "key": "api_key",
            "label": "Azure API Key",
            "field_type": "password",
            "required": True,
        },
    ],
    "OpenAI_Compatible": [
        {
            "key": "api_base",
            "label": "API Base",
            "placeholder": "https://...",
            "required": True,
        },
        {
            "key": "api_key",
            "label": "OpenAI API Key",
            "field_type": "password",
            "required": True,
        },
    ],
    "Dashscope": [
        {
            "key": "api_key",
            "label": "Dashscope API Key",
            "field_type": "password",
            "required": True,
        },
        {
            "key": "api_base",
            "label": "API Base",
            "placeholder": "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
            "default_value": "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
            "required": True,
            "tooltip": "The base URL for your Dashscope server. Defaults to https://dashscope-intl.aliyuncs.com/compatible-mode/v1 if not specified.",
        },
    ],
    "OpenAI_Text_Compatible": [
        {
            "key": "api_base",
            "label": "API Base",
            "placeholder": "https://...",
            "required": True,
        },
        {
            "key": "api_key",
            "label": "OpenAI API Key",
            "field_type": "password",
            "required": True,
        },
    ],
    "Bedrock": [
        {
            "key": "aws_access_key_id",
            "label": "AWS Access Key ID",
            "field_type": "password",
            "tooltip": "You can provide the raw key or the environment variable (e.g. `os.environ/MY_SECRET_KEY`).",
        },
        {
            "key": "aws_secret_access_key",
            "label": "AWS Secret Access Key",
            "field_type": "password",
            "tooltip": "You can provide the raw key or the environment variable (e.g. `os.environ/MY_SECRET_KEY`).",
        },
        {
            "key": "aws_session_token",
            "label": "AWS Session Token",
            "field_type": "password",
            "tooltip": "Temporary credentials session token. You can provide the raw token or the environment variable (e.g. `os.environ/MY_SESSION_TOKEN`).",
        },
        {
            "key": "aws_region_name",
            "label": "AWS Region Name",
            "placeholder": "us-east-1",
            "tooltip": "You can provide the raw key or the environment variable (e.g. `os.environ/MY_SECRET_KEY`).",
        },
        {
            "key": "aws_session_name",
            "label": "AWS Session Name",
            "placeholder": "my-session",
            "tooltip": "Name for the AWS session. You can provide the raw value or the environment variable (e.g. `os.environ/MY_SESSION_NAME`).",
        },
        {
            "key": "aws_profile_name",
            "label": "AWS Profile Name",
            "placeholder": "default",
            "tooltip": "AWS profile name to use for authentication. You can provide the raw value or the environment variable (e.g. `os.environ/MY_PROFILE_NAME`).",
        },
        {
            "key": "aws_role_name",
            "label": "AWS Role Name",
            "placeholder": "MyRole",
            "tooltip": "AWS IAM role name to assume. You can provide the raw value or the environment variable (e.g. `os.environ/MY_ROLE_NAME`).",
        },
        {
            "key": "aws_web_identity_token",
            "label": "AWS Web Identity Token",
            "field_type": "password",
            "tooltip": "Web identity token for OIDC authentication. You can provide the raw token or the environment variable (e.g. `os.environ/MY_WEB_IDENTITY_TOKEN`).",
        },
        {
            "key": "aws_bedrock_runtime_endpoint",
            "label": "AWS Bedrock Runtime Endpoint",
            "placeholder": "https://bedrock-runtime.us-east-1.amazonaws.com",
            "tooltip": "Custom Bedrock runtime endpoint URL. You can provide the raw value or the environment variable (e.g. `os.environ/MY_BEDROCK_ENDPOINT`).",
        },
    ],
    "SageMaker": [
        {
            "key": "aws_access_key_id",
            "label": "AWS Access Key ID",
            "field_type": "password",
            "tooltip": "You can provide the raw key or the environment variable (e.g. `os.environ/MY_SECRET_KEY`).",
        },
        {
            "key": "aws_secret_access_key",
            "label": "AWS Secret Access Key",
            "field_type": "password",
            "tooltip": "You can provide the raw key or the environment variable (e.g. `os.environ/MY_SECRET_KEY`).",
        },
        {
            "key": "aws_region_name",
            "label": "AWS Region Name",
            "placeholder": "us-east-1",
            "tooltip": "You can provide the raw key or the environment variable (e.g. `os.environ/MY_SECRET_KEY`).",
        },
    ],
    "Ollama": [
        {
            "key": "api_base",
            "label": "API Base",
            "placeholder": "http://localhost:11434",
            "default_value": "http://localhost:11434",
            "tooltip": "The base URL for your Ollama server. Defaults to http://localhost:11434 if not specified.",
        },
    ],
    "Anthropic": [
        {
            "key": "api_key",
            "label": "API Key",
            "placeholder": "sk-",
            "field_type": "password",
            "required": True,
        },
    ],
    "Deepgram": [
        {
            "key": "api_key",
            "label": "API Key",
            "field_type": "password",
            "required": True,
        },
    ],
    "ElevenLabs": [
        {
            "key": "api_key",
            "label": "API Key",
            "field_type": "password",
            "required": True,
        },
    ],
    "Google_AI_Studio": [
        {
            "key": "api_key",
            "label": "API Key",
            "placeholder": "aig-",
            "field_type": "password",
            "required": True,
        },
    ],
    "Groq": [
        {
            "key": "api_key",
            "label": "API Key",
            "field_type": "password",
            "required": True,
        },
    ],
    "MistralAI": [
        {
            "key": "api_key",
            "label": "API Key",
            "field_type": "password",
            "required": True,
        },
    ],
    "Deepseek": [
        {
            "key": "api_key",
            "label": "API Key",
            "field_type": "password",
            "required": True,
        },
    ],
    "Cohere": [
        {
            "key": "api_key",
            "label": "API Key",
            "field_type": "password",
            "required": True,
        },
    ],
    "Databricks": [
        {
            "key": "api_key",
            "label": "API Key",
            "field_type": "password",
            "required": True,
        },
    ],
    "xAI": [
        {
            "key": "api_key",
            "label": "API Key",
            "field_type": "password",
            "required": True,
        },
    ],
    "AIML": [
        {
            "key": "api_key",
            "label": "API Key",
            "field_type": "password",
            "required": True,
        },
    ],
    "Cerebras": [
        {
            "key": "api_key",
            "label": "API Key",
            "field_type": "password",
            "required": True,
        },
    ],
    "Sambanova": [
        {
            "key": "api_key",
            "label": "API Key",
            "field_type": "password",
            "required": True,
        },
    ],
    "Perplexity": [
        {
            "key": "api_key",
            "label": "API Key",
            "field_type": "password",
            "required": True,
        },
    ],
    "TogetherAI": [
        {
            "key": "api_key",
            "label": "API Key",
            "field_type": "password",
            "required": True,
        },
    ],
    "Openrouter": [
        {
            "key": "api_key",
            "label": "API Key",
            "field_type": "password",
            "required": True,
        },
    ],
    "FireworksAI": [
        {
            "key": "api_key",
            "label": "API Key",
            "field_type": "password",
            "required": True,
        },
    ],
    "GradientAI": [
        {
            "key": "api_base",
            "label": "GradientAI Endpoint",
            "placeholder": "https://...",
        },
        {
            "key": "api_key",
            "label": "GradientAI API Key",
            "field_type": "password",
            "required": True,
        },
    ],
    "Triton": [
        {
            "key": "api_key",
            "label": "API Key",
            "field_type": "password",
        },
        {
            "key": "api_base",
            "label": "API Base",
            "placeholder": "http://localhost:8000/generate",
        },
    ],
    "Hosted_Vllm": [
        {
            "key": "api_base",
            "label": "API Base",
            "placeholder": "https://...",
            "required": True,
        },
        {
            "key": "api_key",
            "label": "vLLM API Key",
            "field_type": "password",
        },
    ],
    "Voyage": [
        {
            "key": "api_key",
            "label": "API Key",
            "field_type": "password",
            "required": True,
        },
    ],
    "JinaAI": [
        {
            "key": "api_key",
            "label": "API Key",
            "field_type": "password",
            "required": True,
        },
    ],
    "VolcEngine": [
        {
            "key": "api_key",
            "label": "API Key",
            "field_type": "password",
            "required": True,
        },
    ],
    "DeepInfra": [
        {
            "key": "api_key",
            "label": "API Key",
            "field_type": "password",
            "required": True,
        },
    ],
    "Oracle": [
        {
            "key": "api_key",
            "label": "API Key",
            "field_type": "password",
            "required": True,
        },
    ],
    "Snowflake": [
        {
            "key": "api_key",
            "label": "Snowflake API Key / JWT Key for Authentication",
            "field_type": "password",
            "required": True,
        },
        {
            "key": "api_base",
            "label": "Snowflake API Endpoint",
            "placeholder": "https://1234567890.snowflakecomputing.com/api/v2/cortex/inference:complete",
            "tooltip": "Enter the full endpoint with path here. Example: https://1234567890.snowflakecomputing.com/api/v2/cortex/inference:complete",
            "required": True,
        },
    ],
    "Infinity": [
        {
            "key": "api_base",
            "label": "API Base",
            "placeholder": "http://localhost:7997",
        },
    ],
    "FalAI": [
        {
            "key": "api_key",
            "label": "API Key",
            "field_type": "password",
            "required": True,
        },
    ],
}


def _normalize_field(field: Dict[str, Any]) -> ProviderCredentialField:
    return ProviderCredentialField(
        key=field["key"],
        label=field["label"],
        placeholder=field.get("placeholder"),
        tooltip=field.get("tooltip"),
        required=field.get("required", False),
        field_type=field.get("field_type", "text"),
        options=field.get("options"),
        default_value=field.get("default_value"),
    )


def get_provider_create_metadata() -> List[ProviderCreateInfo]:
    providers: List[ProviderCreateInfo] = []

    for provider_key, base_info in PROVIDER_BASE_INFO.items():
        raw_fields = PROVIDER_CREDENTIAL_FIELDS.get(provider_key, _FALLBACK_FIELDS)
        normalized_fields = [_normalize_field(field) for field in raw_fields]

        providers.append(
            ProviderCreateInfo(
                provider=provider_key,
                provider_display_name=base_info["provider_display_name"],
                litellm_provider=base_info["litellm_provider"],
                default_model_placeholder=base_info.get(
                    "default_model_placeholder", DEFAULT_MODEL_PLACEHOLDER
                ),
                credential_fields=normalized_fields,
            )
        )

    # Ensure we have metadata entries for all providers defined in LlmProviders.
    # If a provider enum value is not already present in the litellm_provider
    # field of any entry, create a default entry for it using the fallback
    # credential fields (api_key + api_base) and a generated display name.
    existing_litellm_providers = {p.litellm_provider for p in providers}

    for provider_enum in LlmProviders:
        litellm_provider_value = provider_enum.value
        if litellm_provider_value in existing_litellm_providers:
            continue

        normalized_fields = [_normalize_field(field) for field in _FALLBACK_FIELDS]
        provider_display_name = provider_enum.value.replace("_", " ").title()

        providers.append(
            ProviderCreateInfo(
                provider=provider_enum.name,
                provider_display_name=provider_display_name,
                litellm_provider=litellm_provider_value,
                default_model_placeholder=DEFAULT_MODEL_PLACEHOLDER,
                credential_fields=normalized_fields,
            )
        )

    providers.sort(key=lambda item: item.provider_display_name.lower())
    return providers

