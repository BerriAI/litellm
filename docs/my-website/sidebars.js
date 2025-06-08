/**
 * Creating a sidebar enables you to:
 - create an ordered group of docs
 - render a sidebar for each doc of that group
 - provide next/previous navigation

 The sidebars can be generated from the filesystem, or explicitly defined here.

 Create as many sidebars as you want.
 */

// @ts-check

/** @type {import('@docusaurus/plugin-content-docs').SidebarsConfig} */
const sidebars = {
  // // By default, Docusaurus generates a sidebar from the docs folder structure

  // But you can create a sidebar manually
  tutorialSidebar: [
    { type: "doc", id: "index" }, // NEW
    
    {
      type: "category",
      label: "LiteLLM Proxy Server",
      link: {
        type: "generated-index",
        title: "LiteLLM Proxy Server (LLM Gateway)",
        description: `OpenAI Proxy Server (LLM Gateway) to call 100+ LLMs in a unified interface & track spend, set budgets per virtual key/user`,
        slug: "/simple_proxy",
      },
      items: [
        "proxy/docker_quick_start",
        {
          "type": "category",
          "label": "Config.yaml",
          "items": ["proxy/configs", "proxy/config_management", "proxy/config_settings"]
        },
        {
          type: "category",
          label: "Setup & Deployment",
          items: [
            "proxy/deploy",
            "proxy/prod",
            "proxy/cli",
            "proxy/release_cycle",
            "proxy/model_management",
            "proxy/health",
            "proxy/debugging",
            "proxy/spending_monitoring",
            "proxy/master_key_rotations",
          ],
        },
        "proxy/demo",
        {
          type: "category",
          label: "Architecture",
          items: ["proxy/architecture", "proxy/db_info", "proxy/db_deadlocks", "router_architecture", "proxy/user_management_heirarchy", "proxy/jwt_auth_arch", "proxy/image_handling", "proxy/spend_logs_deletion"],
        },
        {
          type: "link",
          label: "All Endpoints (Swagger)",
          href: "https://litellm-api.up.railway.app/",
        },
        "proxy/enterprise",
        "proxy/management_cli",
        {
          type: "category",
          label: "Making LLM Requests",
          items: [
            "proxy/user_keys",
            "proxy/clientside_auth",
            "proxy/request_headers",
            "proxy/response_headers",
            "proxy/model_discovery",
          ],
        },
        {
          type: "category",
          label: "Authentication",
          items: [
            "proxy/virtual_keys",
            "proxy/token_auth",
            "proxy/service_accounts",
            "proxy/access_control",
            "proxy/custom_auth",
            "proxy/ip_address",
            "proxy/email",
            "proxy/multiple_admins",
          ],
        },
        {
          type: "category",
          label: "Model Access",
          items: [
            "proxy/model_access",
            "proxy/team_model_add"
          ]
        },
        {
          type: "category",
          label: "Admin UI",
          items: [
            "proxy/ui",
            "proxy/admin_ui_sso",
            "proxy/custom_root_ui",
            "proxy/self_serve",
            "proxy/public_teams",
            "tutorials/scim_litellm",
            "proxy/custom_sso",
            "proxy/ui_credentials",
            {
              type: "category",
              label: "UI Logs",
              items: [
                "proxy/ui_logs",
                "proxy/ui_logs_sessions"
              ]
            }
          ],
        },
        {
          type: "category",
          label: "Spend Tracking",
          items: ["proxy/cost_tracking", "proxy/custom_pricing", "proxy/billing",],
        },
        {
          type: "category",
          label: "Budgets + Rate Limits",
          items: ["proxy/users", "proxy/temporary_budget_increase", "proxy/rate_limit_tiers", "proxy/team_budgets", "proxy/customers"],
        },
        {
          type: "link",
          label: "Load Balancing, Routing, Fallbacks",
          href: "https://docs.litellm.ai/docs/routing-load-balancing",
        },
        {
          type: "category",
          label: "Logging, Alerting, Metrics",
          items: [
            "proxy/logging",
            "proxy/logging_spec",
            "proxy/team_logging",
            "proxy/prometheus",
            "proxy/alerting",
            "proxy/pagerduty"],
        },
        {
          type: "category",
          label: "[Beta] Guardrails",
          items: [
            "proxy/guardrails/quick_start",
            ...[
              "proxy/guardrails/aim_security",
              "proxy/guardrails/aporia_api",
              "proxy/guardrails/bedrock",
              "proxy/guardrails/guardrails_ai",
              "proxy/guardrails/lakera_ai",
              "proxy/guardrails/pangea",
              "proxy/guardrails/pii_masking_v2",
              "proxy/guardrails/secret_detection",
              "proxy/guardrails/custom_guardrail",
              "proxy/guardrails/prompt_injection",
            ].sort(),
          ],
        },
        {
          type: "category",
          label: "Secret Managers",
          items: [
            "secret",
            "oidc"
          ]
        },
        {
          type: "category",
          label: "Create Custom Plugins",
          description: "Modify requests, responses, and more",
          items: [
            "proxy/call_hooks",
            "proxy/rules",
          ]
        },
        "proxy/caching",
      ]
    },
    {
      type: "category",
      label: "Supported Endpoints",
      link: {
        type: "generated-index",
        title: "Supported Endpoints",
        description:
          "Learn how to deploy + call models from different providers on LiteLLM",
        slug: "/supported_endpoints",
      },
      items: [
        {
          type: "category",
          label: "/chat/completions",
          link: {
            type: "generated-index",
            title: "Chat Completions",
            description: "Details on the completion() function",
            slug: "/completion",
          },
          items: [
            "completion/input",
            "completion/output",
            "completion/usage",
          ],
        },
        "response_api",
        "text_completion",
        "embedding/supported_embedding",
        "anthropic_unified",
        "mcp",
        {
          type: "category",
          label: "/images",
          items: [
            "image_generation",
            "image_edits",
            "image_variations",
          ]
        },
        {
          type: "category",
          label: "/audio",
          "items": [
            "audio_transcription",
            "text_to_speech",
          ]
        },
        {
          type: "category",
          label: "Pass-through Endpoints (Anthropic SDK, etc.)",
          items: [
            "pass_through/intro",
            "pass_through/vertex_ai",
            "pass_through/google_ai_studio",
            "pass_through/cohere",
            "pass_through/vllm",
            "pass_through/mistral",
            "pass_through/openai_passthrough",
            "pass_through/anthropic_completion",
            "pass_through/bedrock",
            "pass_through/assembly_ai",
            "pass_through/langfuse",
            "proxy/pass_through",
          ],
        },
        "rerank",
        "assistants",

        {
          type: "category",
          label: "/files",
          items: [
            "files_endpoints",
            "proxy/litellm_managed_files",
          ],
        },
        {
          type: "category",
          label: "/batches",
          items: [
            "batches",
            "proxy/managed_batches",
          ]
        },
        "realtime",
        {
          type: "category",
          label: "/fine_tuning",
          items: [
            "fine_tuning",
            "proxy/managed_finetuning",
          ]
        },
        "moderation",
        "apply_guardrail",
      ],
    },
    {
      type: "category",
      label: "Supported Models & Providers",
      link: {
        type: "generated-index",
        title: "Providers",
        description:
          "Learn how to deploy + call models from different providers on LiteLLM",
        slug: "/providers",
      },
      items: [
        {
          type: "category",
          label: "OpenAI",
          items: [
            "providers/openai",
            "providers/openai/responses_api",
            "providers/openai/text_to_speech",
          ]
        },
        "providers/text_completion_openai",
        "providers/openai_compatible",
        {
          type: "category",
          label: "Azure OpenAI",
          items: [
            "providers/azure/azure",
            "providers/azure/azure_embedding",
          ]
        },
        "providers/azure_ai",
        "providers/aiml",
        "providers/vertex",
        {
          type: "category",
          label: "Google AI Studio",
          items: [
            "providers/gemini",
            "providers/google_ai_studio/files",
            "providers/google_ai_studio/realtime",
          ]
        },
        "providers/anthropic",
        "providers/aws_sagemaker",
        {
          type: "category",
          label: "Bedrock",
          items: [
            "providers/bedrock",
            "providers/bedrock_agents",
            "providers/bedrock_vector_store",
          ]
        },
        "providers/litellm_proxy",
        "providers/meta_llama",
        "providers/mistral",
        "providers/codestral",
        "providers/cohere",
        "providers/anyscale",
        {
          type: "category",
          label: "HuggingFace",
          items: [
            "providers/huggingface",
            "providers/huggingface_rerank",
          ]
        },
        "providers/databricks",
        "providers/deepgram",
        "providers/watsonx",
        "providers/predibase",
        "providers/nvidia_nim",
        { type: "doc", id: "providers/nscale", label: "Nscale (EU Sovereign)" },
        "providers/xai",
        "providers/lm_studio",
        "providers/cerebras",
        "providers/volcano",
        "providers/triton-inference-server",
        "providers/ollama",
        "providers/perplexity",
        "providers/friendliai",
        "providers/galadriel",
        "providers/topaz",
        "providers/groq",
        "providers/github",
        "providers/deepseek",
        "providers/fireworks_ai",
        "providers/clarifai",
        "providers/vllm",
        "providers/llamafile",
        "providers/infinity",
        "providers/xinference",
        "providers/cloudflare_workers",
        "providers/deepinfra",
        "providers/ai21",
        "providers/nlp_cloud",
        "providers/replicate",
        "providers/togetherai",
        "providers/novita",
        "providers/voyage",
        "providers/jina_ai",
        "providers/aleph_alpha",
        "providers/baseten",
        "providers/openrouter",
        "providers/sambanova",
        "providers/custom_llm_server",
        "providers/petals",
        "providers/snowflake",
        "providers/featherless_ai",
        "providers/nebius"
      ],
    },
    {
      type: "category",
      label: "Guides",
      items: [
        "exception_mapping",
        "completion/provider_specific_params",
        "guides/finetuned_models",
        "guides/security_settings",
        "completion/audio",
        "completion/web_search",
        "completion/document_understanding",
        "completion/vision",
        "completion/json_mode",
        "reasoning_content",
        "completion/prompt_caching",
        "completion/predict_outputs",
        "completion/knowledgebase",
        "completion/prefix",
        "completion/drop_params",
        "completion/prompt_formatting",
        "completion/stream",
        "completion/message_trimming",
        "completion/function_call",
        "completion/model_alias",
        "completion/batching",
        "completion/mock_requests",
        "completion/reliable_completions",

      ]
    },
    
    {
      type: "category",
      label: "Routing, Loadbalancing & Fallbacks",
      link: {
        type: "generated-index",
        title: "Routing, Loadbalancing & Fallbacks",
        description: "Learn how to load balance, route, and set fallbacks for your LLM requests",
        slug: "/routing-load-balancing",
      },
      items: ["routing", "scheduler", "proxy/load_balancing", "proxy/reliability", "proxy/timeout", "proxy/tag_routing", "proxy/provider_budget_routing", "wildcard_routing"],
    },
    {
      type: "category",
      label: "LiteLLM Python SDK",
      items: [
        "set_keys",
        "completion/token_usage",
        "sdk_custom_pricing",
        "embedding/async_embedding",
        "embedding/moderation",
        "budget_manager",
        "caching/all_caches",
        "migration",
        {
          type: "category",
          label: "LangChain, LlamaIndex, Instructor Integration",
          items: ["langchain/langchain", "tutorials/instructor"],
        },
      ],
    },
    {
      type: "category",
      label: "[Beta] Prompt Management",
      items: [
        "proxy/prompt_management",
        "proxy/custom_prompt_management"
      ],
    },
    {
      type: "category",
      label: "Load Testing",
      items: [
        "benchmarks",
        "load_test_advanced",
        "load_test_sdk",
        "load_test_rpm",
      ]
    },
    {
      type: "category",
      label: "Logging & Observability",
      items: [
        "observability/agentops_integration",
        "observability/langfuse_integration",
        "observability/lunary_integration",
        "observability/deepeval_integration",
        "observability/mlflow",
        "observability/gcs_bucket_integration",
        "observability/langsmith_integration",
        "observability/literalai_integration",
        "observability/opentelemetry_integration",
        "observability/logfire_integration",
        "observability/argilla",
        "observability/arize_integration",
        "observability/phoenix_integration",
        "debugging/local_debugging",
        "observability/raw_request_response",
        "observability/custom_callback",
        "observability/humanloop",
        "observability/scrub_data",
        "observability/braintrust",
        "observability/sentry",
        "observability/lago",
        "observability/helicone_integration",
        "observability/openmeter",
        "observability/promptlayer_integration",
        "observability/wandb_integration",
        "observability/slack_integration",
        "observability/athina_integration",
        "observability/greenscale_integration",
        "observability/supabase_integration",
        `observability/telemetry`,
        "observability/opik_integration",
      ],
    },
    {
      type: "category",
      label: "Tutorials",
      items: [
        "tutorials/openweb_ui",
        "tutorials/openai_codex",
        "tutorials/anthropic_file_usage",
        "tutorials/msft_sso",
        "tutorials/prompt_caching",
        "tutorials/tag_management",
        'tutorials/litellm_proxy_aporia',
        "tutorials/gemini_realtime_with_audio",
        {
          type: "category",
          label: "LiteLLM Python SDK Tutorials",
          items: [
            'tutorials/google_adk',
            'tutorials/azure_openai',
            'tutorials/instructor',
            "tutorials/gradio_integration",
            "tutorials/huggingface_codellama",
            "tutorials/huggingface_tutorial",
            "tutorials/TogetherAI_liteLLM",
            "tutorials/finetuned_chat_gpt",
            "tutorials/text_completion",
            "tutorials/first_playground",
            "tutorials/model_fallbacks",
          ],
        },
      ]
    },
    {
      type: "category",
      label: "Contributing",
      items: [
        "extras/contributing_code",
        {
          type: "category",
          label: "Adding Providers",
          items: [
            "adding_provider/directory_structure",
            "adding_provider/new_rerank_provider"],
        },
        "extras/contributing",
        "contributing",
      ]
    },
    {
      type: "category",
      label: "Extras",
      items: [
        "data_security",
        "data_retention",
        "migration_policy",
        {
          type: "category",
          label: "❤️ 🚅 Projects built on LiteLLM",
          link: {
            type: "generated-index",
            title: "Projects built on LiteLLM",
            description:
              "Learn how to deploy + call models from different providers on LiteLLM",
            slug: "/project",
          },
          items: [
            "projects/smolagents",
            "projects/Docq.AI",
            "projects/PDL",
            "projects/OpenInterpreter",
            "projects/Elroy",
            "projects/dbally",
            "projects/FastREPL",
            "projects/PROMPTMETHEUS",
            "projects/Codium PR Agent",
            "projects/Prompt2Model",
            "projects/SalesGPT",
            "projects/Quivr",
            "projects/Langstream",
            "projects/Otter",
            "projects/GPT Migrate",
            "projects/YiVal",
            "projects/LiteLLM Proxy",
            "projects/llm_cord",
            "projects/pgai",
            "projects/GPTLocalhost",
          ],
        },
        "extras/code_quality",
        "rules",
        "proxy/team_based_routing",
        "proxy/customer_routing",
        "proxy_server",
      ],
    },
    "troubleshoot",
  ],
};

module.exports = sidebars;
