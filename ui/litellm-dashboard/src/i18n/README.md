# Dashboard translations

English remains the default. Choose Chinese on the login page or in the sidebar account menu to translate login and navigation. The shared language setting saves the choice in `localStorage.litellm_language` and reloads the page. Loading the chosen locale before mounting the dashboard keeps React labels, form validation and imperative translations in the same language. If storage is unavailable, the choice cannot persist. Unknown languages and failed locale loads fall back to English.

The Chinese catalog is intentionally partial. Dashboard page bodies, server errors, worker and plugin names, product names and role values remain unchanged or fall back to English. The shared i18next instance retains the upstream number and date formatters. Register locales in `languages.ts` with asynchronous imports, and use the existing nested keys from `locales/en.json`. Keep interpolation names and `Trans` component tags identical across languages.

This replaces the branch's earlier separate provider and `litellm_ui_language` setting. An earlier choice stored only under that key must be selected again. No new dependency, backend service or language-specific route is needed.

Chinese terminology adapts the MIT-licensed locale contributed by [2099383411 in PR #30499](https://github.com/BerriAI/litellm/pull/30499) at `e787b318a932c9c6b897830f6d01045f05bbd0a8`, with additions for the current dashboard. That PR's Ant Design integration is not included here.
