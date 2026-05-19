import type { SidebarsConfig } from "@docusaurus/plugin-content-docs";

const sidebars: SidebarsConfig = {
  main: [
    "intro",
    {
      type: "category",
      label: "Concepts",
      collapsed: false,
      items: [
        "concepts/overview",
        "concepts/models",
        "concepts/agents",
        "concepts/mcps",
        "concepts/skills",
        "concepts/app-tenancy",
      ],
    },
    {
      type: "category",
      label: "Quickstart",
      collapsed: false,
      items: [
        "quickstart/xct-chat",
        "quickstart/xct-home",
        "quickstart/xct-agent-desktop",
      ],
    },
    {
      type: "category",
      label: "Recipes",
      collapsed: true,
      items: [
        "recipes/list-capabilities",
        "recipes/stream-agent",
        "recipes/inject-skill",
        "recipes/use-mcp-tool",
        "recipes/oauth-pkce-react",
        "recipes/handle-budget-exhausted",
        "recipes/subscribe-webhook",
        "recipes/import-marketplace-agent",
        "recipes/upload-skill-zip",
        "recipes/debug-empty-capabilities",
      ],
    },
    {
      type: "category",
      label: "API Reference",
      collapsed: true,
      items: [
        "api-reference/overview",
        "api-reference/openapi-public",
      ],
    },
  ],
};

export default sidebars;
