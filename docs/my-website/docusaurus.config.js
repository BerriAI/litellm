// @ts-check
// Note: type annotations allow type checking and IDEs autocompletion

// @ts-ignore
const lightCodeTheme = require('prism-react-renderer/themes/vsLight');
// @ts-ignore
const darkCodeTheme = require('prism-react-renderer/themes/nightOwl');

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
        onInlineAuthors: 'ignore',
        onUntruncatedBlogPosts: 'ignore',
      },
    ],
    [
      '@docusaurus/plugin-content-blog',
      {
        id: 'blog',
        path: './blog',
        routeBasePath: 'blog',
        blogTitle: 'Blog',
        blogSidebarTitle: 'All Posts',
        blogSidebarCount: 'ALL',
        postsPerPage: 10,
        showReadingTime: false,
        sortPosts: 'descending',
        include: ['**/index.{md,mdx}'],
      },
    ],

    [
      '@scalar/docusaurus',
      {
        label: 'API Reference',
        route: '/api-reference',
        showNavLink: false,
        configuration: {
          url: '/openapi.json',
          title: 'LiteLLM API Reference',
          theme: 'default',
        },
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
    // Shim that ensures window.gtag is always defined, preventing
    // "window.gtag is not a function" errors when GA script hasn't loaded yet
    () => ({
      name: 'gtag-shim',
      injectHtmlTags() {
        return {
          headTags: [
            {
              tagName: 'script',
              innerHTML: `window.dataLayer=window.dataLayer||[];function gtag(){dataLayer.push(arguments);}if(!window.gtag){window.gtag=gtag;}`,
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
        gtag: process.env.NODE_ENV === 'production'
          ? { trackingID: 'G-K7K215ZVNC', anonymizeIP: true }
          : undefined,
        docs: {
          sidebarPath: require.resolve('./sidebars.js'),
        },
        blog: false, // Disable the default blog plugin from preset-classic
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
    },
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
            type: 'docSidebar',
            sidebarId: 'learnSidebar',
            position: 'left',
            label: 'Learn',
          },
          {
            type: 'docSidebar',
            sidebarId: 'integrationsSidebar',
            position: 'left',
            label: 'Integrations',
          },
          {
            position: 'left',
            label: 'Enterprise',
            to: "docs/enterprise"
          },
          { to: '/blog', label: 'Blog', position: 'left' },
          { to: '/api-reference', label: 'API Reference', position: 'left' },
          {
            href: 'https://models.litellm.ai/',
            label: '💸 Cost Map',
            position: 'right',
          },
          {
            href: 'https://github.com/BerriAI/litellm',
            position: 'right',
            className: 'header-github-link',
            'aria-label': 'GitHub repository',
          },
          {
            href: 'https://www.litellm.ai/support',
            position: 'right',
            className: 'header-discord-link',
            'aria-label': 'Discord / Slack community',
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
