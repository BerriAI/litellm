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
    "tutorials/first_playground",
    {
      type: "category",
      label: "Completion()",
      items: ["completion/input", "completion/output", "completion/reliable_completions"],
    },
    {
      type: "category",
      label: "Embedding()",
      items: ["embedding/supported_embedding"],
    },
    'completion/supported',
    "token_usage",
    "exception_mapping",
    "stream",
    'debugging/hosted_debugging',
    'debugging/local_debugging',
    {
      type: 'category',
      label: 'Tutorials',
      items: [
        'tutorials/huggingface_tutorial', 
        'tutorials/TogetherAI_liteLLM', 
        'tutorials/fallbacks', 
        'tutorials/finetuned_chat_gpt',
        'tutorials/ab_test_llms'
      ],
    },
    {
      type: "category",
      label: "Logging & Observability",
      items: [
        "observability/callbacks",
        "observability/integrations",
        "observability/promptlayer_integration",
        "observability/llmonitor_integration",
        "observability/helicone_integration",
        "observability/supabase_integration",
      ],
    },
    {
      type: 'category',
      label: 'Extras',
      items: [
        'extras/secret', 
        'extras/caching', 
      ],
    },
    "troubleshoot",
    "contributing",
    "contact",
  ],
};

module.exports = sidebars;
