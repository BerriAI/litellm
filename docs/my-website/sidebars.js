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
            "proxy/model_management",
            "proxy/health",
            "proxy/debugging",
            "proxy/pass_through",
        ],
        },
        "proxy/demo",
        {
          type: "category",
          label: "Architecture",
          items: ["proxy/architecture", "proxy/db_info", "router_architecture"],
        }, 
        {
          type: "link",
          label: "All Endpoints (Swagger)",
          href: "https://litellm-api.up.railway.app/",
        },
        "proxy/enterprise",
        {
          type: "category",
          label: "Making LLM Requests",
          items: [
            "proxy/user_keys",
            "proxy/response_headers", 
            "pass_through/vertex_ai",
            "pass_through/google_ai_studio",
            "pass_through/cohere",
            "pass_through/anthropic_completion",
            "pass_through/bedrock",
            "pass_through/langfuse"
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
            "proxy/ip_address",
            "proxy/email",
            "proxy/multiple_admins",
          ],
        },
        {
          type: "category",
          label: "Admin UI",
          items: [
            "proxy/ui", 
            "proxy/self_serve", 
            "proxy/custom_sso"
          ],
        },
        {
          type: "category",
          label: "Spend Tracking + Budgets",
          items: ["proxy/cost_tracking", "proxy/users", "proxy/custom_pricing", "proxy/team_budgets", "proxy/billing", "proxy/customers"],
        },
        {
          type: "link",
          label: "Load Balancing, Routing, Fallbacks",
          href: "https://docs.litellm.ai/docs/routing-load-balancing",
        },
        {
          type: "category",
          label: "Logging, Alerting, Metrics",
          items: ["proxy/logging", "proxy/team_logging","proxy/alerting", "proxy/prometheus",],
        },
        {
          type: "category",
          label: "[Beta] Guardrails",
          items: [
            "proxy/guardrails/quick_start", 
            "proxy/guardrails/aporia_api", 
            "proxy/guardrails/guardrails_ai",
            "proxy/guardrails/lakera_ai", 
            "proxy/guardrails/bedrock",  
            "proxy/guardrails/pii_masking_v2", 
            "proxy/guardrails/secret_detection", 
            "proxy/guardrails/custom_guardrail", 
            "prompt_injection"
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
        "proxy/caching",
        "proxy/call_hooks",
        "proxy/rules", 
      ]
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
        "providers/openai", 
        "providers/text_completion_openai",
        "providers/openai_compatible",
        "providers/azure", 
        "providers/azure_ai", 
        "providers/vertex", 
        "providers/gemini", 
        "providers/anthropic", 
        "providers/aws_sagemaker",
        "providers/bedrock", 
        "providers/litellm_proxy", 
        "providers/mistral", 
        "providers/codestral",
        "providers/cohere", 
        "providers/anyscale",
        "providers/huggingface", 
        "providers/databricks",
        "providers/watsonx",
        "providers/predibase",
        "providers/nvidia_nim", 
        "providers/xai",
        "providers/lm_studio",
        "providers/cerebras", 
        "providers/volcano", 
        "providers/triton-inference-server",
        "providers/ollama", 
        "providers/perplexity", 
        "providers/friendliai",
        "providers/galadriel",
        "providers/groq", 
        "providers/github", 
        "providers/deepseek", 
        "providers/fireworks_ai",
        "providers/clarifai", 
        "providers/vllm", 
        "providers/xinference", 
        "providers/cloudflare_workers", 
        "providers/deepinfra",
        "providers/ai21", 
        "providers/nlp_cloud",
        "providers/replicate", 
        "providers/togetherai", 
        "providers/voyage", 
        "providers/jina_ai", 
        "providers/aleph_alpha", 
        "providers/baseten", 
        "providers/openrouter", 
        "providers/palm", 
        "providers/sambanova", 
        "providers/custom_llm_server",
        "providers/petals",
        
      ],
    },
    {
      type: "category",
      label: "Guides",
      items: [
        "exception_mapping",
        "completion/provider_specific_params",
        "guides/finetuned_models",
        "completion/audio",
        "completion/document_understanding",
        "completion/vision",
        "completion/json_mode",
        "completion/prompt_caching",
        "completion/predict_outputs",
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
      label: "Supported Endpoints",
      items: [
        {
          type: "category",
          label: "Chat",
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
        "text_completion",
        "embedding/supported_embedding",
        "image_generation",
        {
          type: "category",
          label: "Audio",
          "items": [
            "audio_transcription",
            "text_to_speech",
          ]
        },
        "rerank",
        "assistants",
        "batches",
        "realtime",
        "fine_tuning",
        "moderation",
        {
          type: "link",
          label: "Use LiteLLM Proxy with Vertex, Bedrock SDK",
          href: "/docs/pass_through/vertex_ai",
        },
      ],
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
      items: ["routing", "scheduler", "proxy/load_balancing", "proxy/reliability", "proxy/tag_routing", "proxy/provider_budget_routing", "proxy/team_based_routing", "proxy/customer_routing", "wildcard_routing"],
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
      label: "Load Testing",
      items: [
        "benchmarks",
        "load_test",
        "load_test_advanced",
        "load_test_sdk",
        "load_test_rpm",
      ]
    },
    {
      type: "category",
      label: "Logging & Observability",
      items: [
        "observability/langfuse_integration",
        "observability/gcs_bucket_integration",
        "observability/langsmith_integration",
        "observability/literalai_integration",
        "observability/opentelemetry_integration",
        "observability/logfire_integration",
        "observability/argilla",
        "observability/arize_integration",
        "debugging/local_debugging",
        "observability/raw_request_response",
        "observability/custom_callback",
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
        "observability/lunary_integration",
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
        'tutorials/litellm_proxy_aporia',
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
    {
      type: "category",
      label: "Extras",
      items: [
        "extras/contributing",
        "data_security",
        "migration_policy",
        "contributing",
        "proxy/pii_masking",
        "extras/code_quality",
        "rules",
        "proxy_server",
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
            "projects/Docq.AI",
            "projects/OpenInterpreter",
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
          ],
        },
      ],
    },
    "troubleshoot",
  ],
};

module.exports = sidebars;
