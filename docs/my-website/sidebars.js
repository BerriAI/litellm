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
        "completion/reliable_completions", 
        "completion/multiple_deployments",
        "completion/config",
        "completion/batching",
        "completion/mock_requests",
      ],
    },
    {
      type: "category",
      label: "Embedding() & Moderation()",
      items: [
        "embedding/supported_embedding", 
        "embedding/async_embedding",
        "embedding/moderation",
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
        "providers/azure", 
        "providers/huggingface", 
        "providers/ollama", 
        "providers/vertex", 
        "providers/palm", 
        "providers/anthropic", 
        "providers/aws_sagemaker",
        "providers/bedrock", 
        "providers/anyscale",
        "providers/perplexity", 
        "providers/vllm", 
        "providers/deepinfra",
        "providers/ai21", 
        "providers/nlp_cloud",
        "providers/replicate", 
        "providers/cohere", 
        "providers/togetherai", 
        "providers/aleph_alpha", 
        "providers/baseten", 
        "providers/openrouter", 
        "providers/custom_openai_proxy",
        "providers/petals",
      ]
    },
    "simple_proxy",
    "budget_manager",
    "routing",
    "set_keys",
    "completion/token_usage",
    {
      type: 'category',
      label: 'Tutorials',
      items: [
        'tutorials/azure_openai',
        'tutorials/ab_test_llms',
        'tutorials/oobabooga',
        "tutorials/gradio_integration",
        "tutorials/model_config_proxy",
        'tutorials/huggingface_codellama',
        'tutorials/huggingface_tutorial', 
        'tutorials/TogetherAI_liteLLM', 
        'tutorials/finetuned_chat_gpt',
        'tutorials/sagemaker_llms',
        'tutorials/text_completion',
        'tutorials/litellm_Test_Multiple_Providers',
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
        "observability/sentry",
        "observability/promptlayer_integration",
        "observability/wandb_integration",
        "observability/langfuse_integration",
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
        "caching/caching_api",
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
