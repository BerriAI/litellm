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
    { type: "doc", id: "index" },  // NEW
    {
      type: 'category',
      label: 'Completion()',
      items: ['completion/input','completion/output'],
    },
    {
      type: 'category',
      label: 'Embedding()',
      items: ['embedding/supported_embedding'],
    },
    'completion/supported',
    'debugging/local_debugging',
    'debugging/hosted_debugging',
    {
      type: 'category',
      label: 'Tutorials',
      items: ['tutorials/huggingface_tutorial', 'tutorials/TogetherAI_liteLLM'],
    },
    'token_usage',
    'stream',
    'secret',
    'caching',
    {
      type: 'category',
      label: 'Logging & Observability',
      items: ['observability/callbacks', 'observability/integrations', 'observability/helicone_integration', 'observability/supabase_integration'],
    },
    'troubleshoot',
    'contributing',
    'contact'
  ],
};

module.exports = sidebars;
