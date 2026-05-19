import type { Config } from "@docusaurus/types";
import type * as Preset from "@docusaurus/preset-classic";

const config: Config = {
  title: "xct-litellm",
  tagline: "Capability provider for the XCT ecosystem",
  favicon: "img/favicon.ico",

  url: "https://docs.xct.ai",
  baseUrl: "/",

  organizationName: "XcityUS",
  projectName: "xcity-litellm",

  onBrokenLinks: "warn",
  onBrokenMarkdownLinks: "warn",

  i18n: {
    defaultLocale: "en",
    locales: ["en", "zh-CN"],
  },

  presets: [
    [
      "classic",
      {
        docs: {
          sidebarPath: "./sidebars.ts",
          editUrl: "https://github.com/XcityUS/xcity-litellm/edit/litellm_internal_staging/xct-docs/",
          routeBasePath: "/",
        },
        blog: false,
        theme: {
          customCss: "./src/css/custom.css",
        },
      } satisfies Preset.Options,
    ],
  ],

  themeConfig: {
    navbar: {
      title: "xct-litellm",
      logo: { alt: "XCT logo", src: "img/logo.svg" },
      items: [
        { type: "docSidebar", sidebarId: "main", position: "left", label: "Docs" },
        {
          href: "https://github.com/XcityUS/xcity-litellm",
          label: "GitHub",
          position: "right",
        },
      ],
    },
    footer: {
      style: "dark",
      links: [
        {
          title: "Docs",
          items: [
            { label: "Concepts", to: "/concepts/overview" },
            { label: "Quickstart", to: "/quickstart/xct-chat" },
            { label: "API Reference", to: "/api-reference/overview" },
          ],
        },
        {
          title: "More",
          items: [
            { label: "Source", href: "https://github.com/XcityUS/xcity-litellm" },
            { label: "Upstream", href: "https://docs.litellm.ai" },
          ],
        },
      ],
      copyright: `Copyright © ${new Date().getFullYear()} XcityUS.`,
    },
    prism: {
      additionalLanguages: ["bash", "json", "yaml", "python"],
    },
  } satisfies Preset.ThemeConfig,
};

export default config;
