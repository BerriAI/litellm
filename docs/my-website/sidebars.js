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
    'tutorials/compare_llms',
    {
      type: "category",
      label: "Completion()",
      items: ["completion/input", "completion/output", "completion/model_alias", "completion/reliable_completions", "completion/stream"],
    },
    {
      type: "category",
      label: "Embedding()",
      items: ["embedding/supported_embedding"],
    },
    'completion/supported',
    // {
    //   type: "category",
    //   label: "Providers",
    //   link: {
    //     type: 'generated-index',
    //     title: 'Providers',
    //     description: 'Learn how to deploy + call models from different providers on LiteLLM',
    //     slug: '/providers',
    //   },
    //   items: ["providers/huggingface"],
    // },
    "token_usage",
    "exception_mapping",
    'debugging/local_debugging',
    {
      type: 'category',
      label: 'Tutorials',
      items: [
        "tutorials/model_fallbacks",
        'tutorials/ab_test_llms',
        'tutorials/huggingface_tutorial', 
        'tutorials/TogetherAI_liteLLM', 
        'tutorials/finetuned_chat_gpt',
        'tutorials/text_completion',
        'tutorials/litellm_Test_Multiple_Providers',
        "tutorials/first_playground",
      ],
    },
    {
      type: "category",
      label: "Logging & Observability",
      items: [
        "observability/callbacks",
        "observability/integrations",
        "observability/promptlayer_integration",
        "observability/langfuse_integration",
        "observability/traceloop_integration",
        "observability/llmonitor_integration",
        "observability/helicone_integration",
        "observability/supabase_integration",
      ],
    },
    {
      type: "category",
      label: "Caching",
      items: [
        "caching/caching",
        "caching/gpt_cache",
      ],
    },
    {
      type: 'category',
      label: 'Extras',
      items: [
        'extras/secret',
        'extras/contributing',
      ],
    },
    "projects",
    "troubleshoot",
    "contact",
  ],
};

module.exports = sidebars;
