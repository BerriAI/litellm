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
      label: "💥 LiteLLM Proxy Server",
      link: {
        type: "generated-index",
        title: "💥 LiteLLM Proxy Server (LLM Gateway)",
        description: `OpenAI Proxy Server (LLM Gateway) to call 100+ LLMs in a unified interface & track spend, set budgets per virtual key/user`,
        slug: "/simple_proxy",
      },
      items: [
        "proxy/quick_start",
        "proxy/docker_quick_start",
        "proxy/deploy", 
        "proxy/demo",
        "proxy/prod",
        {
          type: "category",
          label: "Architecture",
          items: ["proxy/architecture"],
        }, 
        {
          type: "link",
          label: "📖 All Endpoints (Swagger)",
          href: "https://litellm-api.up.railway.app/",
        },
        "proxy/enterprise",
        "proxy/user_keys",
        "proxy/configs",
        "proxy/response_headers", 
        "proxy/reliability",
        {
          type: "category",
          label: "🔑 Authentication",
          items: ["proxy/virtual_keys", "proxy/token_auth", "proxy/service_accounts", "proxy/ip_address"],
        },
        {
          type: "category",
          label: "💸 Spend Tracking + Budgets",
          items: ["proxy/cost_tracking", "proxy/users", "proxy/custom_pricing", "proxy/team_budgets", "proxy/billing", "proxy/customers"],
        },
        {
          type: "category",
          label: "Routing",
          items: ["proxy/load_balancing", "proxy/tag_routing", "proxy/team_based_routing", "proxy/customer_routing",],
        },
        {
          type: "category",
          label: "Use with Provider SDKs",
          items: [
            "pass_through/vertex_ai",
            "pass_through/google_ai_studio",
            "pass_through/cohere",
            "anthropic_completion",
            "pass_through/bedrock",
            "pass_through/langfuse"
          ],
        },
        {
          type: "category",
          label: "Admin UI",
          items: ["proxy/ui", "proxy/self_serve", "proxy/custom_sso"],
        },
        {
          type: "category",
          label: "🪢 Logging, Alerting, Metrics",
          items: ["proxy/logging", "proxy/bucket", "proxy/team_logging","proxy/streaming_logging", "proxy/alerting", "proxy/prometheus",],
        },
        {
          type: "category",
          label: "🛡️ [Beta] Guardrails",
          items: [
            "proxy/guardrails/quick_start", 
            "proxy/guardrails/aporia_api", 
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
          label: "Secret Manager - storing LLM API Keys", 
          items: [
            "secret", 
            "oidc"
          ]
        },
        "proxy/caching",
        "proxy/pass_through",
        "proxy/email",
        "proxy/multiple_admins",
        "proxy/model_management",
        "proxy/health",
        "proxy/debugging",
        "proxy/call_hooks",
        "proxy/rules",
        "proxy/cli", 
      ]
    },
    {
      type: "category",
      label: "💯 Supported Models & Providers",
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
        "providers/cerebras", 
        "providers/volcano", 
        "providers/triton-inference-server",
        "providers/ollama", 
        "providers/perplexity", 
        "providers/friendliai",
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
        "providers/aleph_alpha", 
        "providers/baseten", 
        "providers/openrouter", 
        "providers/palm", 
        "providers/sambanova", 
        // "providers/custom_openai_proxy",
        "providers/custom_llm_server",
        "providers/petals",
        
      ],
    },
    {
      type: "category",
      label: "Chat Completions (litellm.completion + PROXY)",
      link: {
        type: "generated-index",
        title: "Chat Completions",
        description: "Details on the completion() function",
        slug: "/completion",
      },
      items: [
        "completion/input",
        "completion/provider_specific_params",
        "completion/json_mode",
        "completion/prefix",
        "completion/drop_params",
        "completion/prompt_formatting",
        "completion/output",
        "completion/prompt_caching",
        "completion/usage",
        "exception_mapping",
        "completion/stream",
        "completion/message_trimming",
        "completion/function_call",
        "completion/vision",
        "completion/model_alias",
        "completion/batching",
        "completion/mock_requests",
        "completion/reliable_completions",
      ],
    },
    {
      type: "category",
      label: "Supported Endpoints - /images, /audio/speech, /assistants etc",
      items: [
        "embedding/supported_embedding",
        "image_generation",
        "audio_transcription",
        "text_to_speech",
        "rerank",
        "assistants",
        "batches",
        "realtime",
        "fine_tuning",
        {
          type: "link",
          label: "Use LiteLLM Proxy with Vertex, Bedrock SDK",
          href: "/docs/pass_through/vertex_ai",
        },
      ],
    },
    "routing",
    "scheduler",
    {
      type: "category",
      label: "🚅 LiteLLM Python SDK",
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
        "observability/opentelemetry_integration",
        "observability/logfire_integration",
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
