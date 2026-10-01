# Dashboard translations

The dashboard defaults to English. Select 简体中文 from the language menu on the login page or dashboard header to switch to Simplified Chinese. The explicit choice is saved in `localStorage.litellm_ui_language`. Unsupported stored values fall back to English. When browser storage is unavailable, the selection works until the page reloads

This initial coverage includes login and validation, the sidebar and collapsed navigation labels, breadcrumbs, view switching, account controls, and header controls. Dashboard page bodies, server error messages, worker and plugin names, and role labels are outside this change

Translations are bundled with the static export using i18next and react-i18next. Each mounted `LanguageProvider` owns its instance, starts in English for server rendering and hydration, then restores the stored preference after mounting. Changing language updates the HTML `lang` attribute without changing URLs or remounting the form. No translation service or language-specific backend is required

Use `useTranslation` from `@/i18n/useTranslation` for React components. It uses the root provider, with an isolated English fallback for standalone components. Add the same key to `locales/en.json` and `locales/zh-CN.json`. Plain display text serves as its own key; longer sentences with markup use descriptive keys and `Trans`. Key and namespace separators are disabled, so punctuation in display text stays literal

Translate labels at render time. Keep identifiers, URLs, form field names, permission checks, plugin names, and API payloads unchanged. Use interpolation for user values and `Trans` for inline links or code, rather than joining translated fragments. React escapes interpolated display values

Chinese terminology adapts the MIT-licensed locale contributed by [2099383411 in PR #30499](https://github.com/BerriAI/litellm/pull/30499) at `e787b318a932c9c6b897830f6d01045f05bbd0a8`, with additions for the current Base UI dashboard. That PR's Ant Design integration is not included here
