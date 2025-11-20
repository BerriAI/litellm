// @ts-check
// Note: type annotations allow type checking and IDEs autocompletion

// @ts-ignore
const lightCodeTheme = require('prism-react-renderer/themes/github');
// @ts-ignore
const darkCodeTheme = require('prism-react-renderer/themes/dracula');

const inkeepConfig = {
  baseSettings: {
    apiKey: "0cb9c9916ec71bfe0e53c9d7f83ff046daee3fa9ef318f6a",
    organizationDisplayName: 'liteLLM',
    primaryBrandColor: '#4965f5',
    theme: {
      styles: [
        {
          key: "custom-theme",
          type: "style",
          value: `
            .ikp-chat-button__button {
              margin-right: 80px !important;
            }
          `,
        },
      ],
      syntaxHighlighter: {
        lightTheme: lightCodeTheme,
        darkTheme: darkCodeTheme,
      },
    },
  },
  searchSettings: {
    searchBarPlaceholder: 'Search docs...',
  },
  aiChatSettings: {
    quickQuestions: [
      'How do I use the proxy?',
      'How do I cache responses?',
      'How do I stream responses?',
    ],
    aiAssistantAvatar: '/img/favicon.ico',
  },
};

/** @type {import('@docusaurus/types').Config} */
const config = {
  title: 'liteLLM',
  tagline: 'Simplify LLM API Calls',
  favicon: '/img/favicon.ico', 

  // Set the production url of your site here
  url: 'https://docs.litellm.ai/',
  // Set the /<baseUrl>/ pathname under which your site is served
  // For GitHub pages deployment, it is often '/<projectName>/'
  baseUrl: '/',

  onBrokenLinks: 'warn',
  onBrokenMarkdownLinks: 'warn',

  // Even if you don't use internalization, you can use this field to set useful
  // metadata like html lang. For example, if your site is Chinese, you may want
  // to replace "en" with "zh-Hans".
  i18n: {
    defaultLocale: 'en',
    locales: ['en'],
  },
  plugins: [
    [
      '@inkeep/cxkit-docusaurus',
      {
        SearchBar: {
          ...inkeepConfig,
        },
        ChatButton: {
          ...inkeepConfig,
        },
      },
    ],
    [
      '@docusaurus/plugin-ideal-image',
      {
        quality: 100,
        max: 1920, // max resized image's size.
        min: 640, // min resized image's size. if original is lower, use that size.
        steps: 2, // the max number of images generated between min and max (inclusive)
        disableInDev: false,
      },
    ],
    [
      '@docusaurus/plugin-content-blog',
      {
        id: 'release_notes',
        path: './release_notes',
        routeBasePath: 'release_notes',
        blogTitle: 'Release Notes',
        blogSidebarTitle: 'Releases',
        blogSidebarCount: 'ALL',
        postsPerPage: 'ALL',
        showReadingTime: false,
        sortPosts: 'descending',
        include: ['**/*.{md,mdx}'],
      },
    ],

    () => ({
      name: 'cripchat',
      injectHtmlTags() {
        return {
          headTags: [
            {
              tagName: 'script',
              innerHTML: `window.$crisp=[];window.CRISP_WEBSITE_ID="be07a4d6-dba0-4df7-961d-9302c86b7ebc";(function(){d=document;s=d.createElement("script");s.src="https://client.crisp.chat/l.js";s.async=1;d.getElementsByTagName("head")[0].appendChild(s);})();`,
            },
          ],
        };
      },
    }),
  ],

  presets: [
    [
      'classic',
      /** @type {import('@docusaurus/preset-classic').Options} */
      ({
        gtag: {
          trackingID: 'G-K7K215ZVNC',
          anonymizeIP: true,
        },
        docs: {
          sidebarPath: require.resolve('./sidebars.js'),
        },
        theme: {
          customCss: require.resolve('./src/css/custom.css'),
        },
      }),
    ],
  ],

  themes: ['@docusaurus/theme-mermaid'],
  markdown: {
    mermaid: true,
  },

  scripts: [
    {
      async: true,
      src: 'https://www.feedbackrocket.io/sdk/v1.2.js',
      'data-fr-id': 'GQwepB0f0L-x_ZH63kR_V',
      'data-fr-theme': 'dynamic',
    }
  ],

  themeConfig:
    /** @type {import('@docusaurus/preset-classic').ThemeConfig} */
    ({
      // Replace with your project's social card
      image: 'img/docusaurus-social-card.png',
      navbar: {
        title: '🚅 LiteLLM',
        items: [
          {
            type: 'docSidebar',
            sidebarId: 'tutorialSidebar',
            position: 'left',
            label: 'Docs',
          },
          {
            sidebarId: 'integrationsSidebar',
            position: 'left',
            label: 'Integrations',
            to: "docs/integrations"
          },
          {
            sidebarId: 'tutorialSidebar',
            position: 'left',
            label: 'Enterprise',
            to: "docs/enterprise"
          },
          { to: '/release_notes', label: 'Release Notes', position: 'left' },
          {
            href: 'https://models.litellm.ai/',
            label: '💸 LLM Model Cost Map',
            position: 'right',
          },
          {
            href: 'https://github.com/BerriAI/litellm',
            label: 'GitHub',
            position: 'right',
          },
          {
            href: 'https://www.litellm.ai/support',
            label: 'Slack/Discord',
            position: 'right',
          }
        ],
      },
      footer: {
        style: 'dark',
        links: [
          {
            title: 'Docs',
            items: [
              {
                label: 'Getting Started',
                to: 'https://docs.litellm.ai/docs/',
              },
            ],
          },
          {
            title: 'Community',
            items: [
              {
                label: 'Discord',
                href: 'https://discord.com/invite/wuPM9dRgDw',
              },
              {
                label: 'Twitter',
                href: 'https://twitter.com/LiteLLM',
              },
            ],
          },
          {
            title: 'More',
            items: [
              {
                label: 'GitHub',
                href: 'https://github.com/BerriAI/litellm/',
              },
            ],
          },
        ],
        copyright: `Copyright © ${new Date().getFullYear()} liteLLM`,
      },
      colorMode: {
        defaultMode: 'light',
        disableSwitch: false,
        respectPrefersColorScheme: true,
      },
      prism: {
        theme: lightCodeTheme,
        darkTheme: darkCodeTheme,
      },
    }),
};

module.exports = config;
