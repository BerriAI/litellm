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
      label: "Completion()",
      link: {
        type: 'generated-index',
        title: 'Completion()',
        description: 'Details on the completion() function',
        slug: '/completion',
      },
      items: [
        "completion/input", 
        "completion/prompt_formatting",
        "completion/output", 
        "exception_mapping",
        "completion/stream", 
        "completion/message_trimming",
        "completion/function_call",
        "completion/model_alias", 
        "completion/batching",
        "completion/mock_requests",
        "completion/reliable_completions",
      ],
    },
    {
      type: "category",
      label: "Embedding(), Moderation(), Image Generation()",
      items: [
        "embedding/supported_embedding", 
        "embedding/async_embedding",
        "embedding/moderation",
        "image_generation"
      ],
    },
    {
      type: "category",
      label: "Supported Models & Providers",
      link: {
        type: 'generated-index',
        title: 'Providers',
        description: 'Learn how to deploy + call models from different providers on LiteLLM',
        slug: '/providers',
      },
      items: [
        "providers/openai", 
        "providers/openai_compatible",
        "providers/azure", 
        "providers/huggingface", 
        "providers/ollama", 
        "providers/vertex", 
        "providers/palm", 
        "providers/mistral", 
        "providers/anthropic", 
        "providers/aws_sagemaker",
        "providers/bedrock", 
        "providers/anyscale",
        "providers/perplexity", 
        "providers/vllm", 
        "providers/cloudflare_workers", 
        "providers/deepinfra",
        "providers/ai21", 
        "providers/nlp_cloud",
        "providers/replicate", 
        "providers/cohere", 
        "providers/togetherai", 
        "providers/voyage", 
        "providers/aleph_alpha", 
        "providers/baseten", 
        "providers/openrouter", 
        "providers/custom_openai_proxy",
        "providers/petals",
      ]
    },
    {
      type: "category",
      label: "💥 OpenAI Proxy Server",
      link: {
        type: 'generated-index',
        title: '💥 OpenAI Proxy Server',
        description: `Proxy Server to call 100+ LLMs in a unified interface, load balance deployments, track costs per user`,
        slug: '/simple_proxy',
      },
      items: [
        "proxy/quick_start", 
        "proxy/configs",
        "proxy/user_keys",
        "proxy/load_balancing", 
        "proxy/virtual_keys",
        "proxy/users",
        "proxy/model_management",
        "proxy/reliability",
        "proxy/health",
        "proxy/call_hooks",
        "proxy/caching",
        "proxy/streaming_logging",
        "proxy/logging", 
        "proxy/cli", 
        "proxy/deploy", 
      ]
    },
    "routing",
    "rules",
    "set_keys",
    "budget_manager",
    "secret",
    "completion/token_usage",
    "load_test",
    {
      type: 'category',
      label: 'Tutorials',
      items: [
        'tutorials/azure_openai',
        "tutorials/lm_evaluation_harness",
        "tutorials/eval_suites",
        'tutorials/oobabooga',
        "tutorials/gradio_integration",
        'tutorials/huggingface_codellama',
        'tutorials/huggingface_tutorial', 
        'tutorials/TogetherAI_liteLLM', 
        'tutorials/finetuned_chat_gpt',
        'tutorials/sagemaker_llms',
        'tutorials/text_completion',
        "tutorials/first_playground",
        'tutorials/compare_llms',
        "tutorials/model_fallbacks",
      ],
    },
    {
      type: "category",
      label: "Logging & Observability",
      items: [
        'debugging/local_debugging',
        "observability/callbacks",
        "observability/custom_callback",
        "observability/langfuse_integration",
        "observability/sentry",
        "observability/promptlayer_integration",
        "observability/wandb_integration",
        "observability/langsmith_integration",
        "observability/slack_integration",
        "observability/traceloop_integration",
        "observability/llmonitor_integration",
        "observability/helicone_integration",
        "observability/supabase_integration",
        `observability/telemetry`,
      ],
    },
    {
      type: "category",
      label: "Caching",
      link: {
        type: 'generated-index',
        title: 'Providers',
        description: 'Learn how to deploy + call models from different providers on LiteLLM',
        slug: '/caching',
      },
      items: [
        "caching/local_caching",
        "caching/redis_cache",
      ],
    },
    {
      type: "category",
      label: "LangChain, LlamaIndex Integration",
      items: [
        "langchain/langchain"
      ],
    },
    {
      type: 'category',
      label: 'Extras',
      items: [
        'extras/contributing',
        "proxy_server",
        {
          type: "category",
          label: "❤️ 🚅 Projects built on LiteLLM",
          link: {
            type: 'generated-index',
            title: 'Projects built on LiteLLM',
            description: 'Learn how to deploy + call models from different providers on LiteLLM',
            slug: '/project',
          },
          items: [
            "projects/Docq.AI",
            "projects/OpenInterpreter",
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
          ]
        },
      ],
    },
    "migration",
    "troubleshoot",
  ],
};

module.exports = sidebars;
