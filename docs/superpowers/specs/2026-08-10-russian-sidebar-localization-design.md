# Russian Sidebar Localization Design

**Date:** 2026-08-10
**Status:** Approved for planning

## Goal

Add Russian as a second LiteLLM Dashboard language without replacing English. The first delivery localizes the AI Gateway sidebar and its account menu, lets every user change the language from the dashboard header, and deploys the locally built LiteLLM image to the existing VPS.

## Scope

### Included

- English and Russian locale resources
- Browser-language detection for users who have not made an explicit choice
- A persisted per-browser language preference
- A language selector in the right side of the dashboard header, available to every authenticated role
- Immediate language changes without a page reload
- Russian translations for:
  - sidebar section headings
  - top-level navigation items
  - nested navigation items
  - collapsed-sidebar tooltips and accessibility labels
  - sidebar account menu labels, controls, and logout action
- English fallback for every missing translation
- Focused localization and sidebar tests
- A custom `linux/amd64` LiteLLM image built from the local clone
- Replacement of only the LiteLLM container on the VPS, preserving PostgreSQL, Caddy, volumes, configuration, and certificates
- Health, smoke, and browser verification after deployment
- A documented rollback to the previously deployed LiteLLM image

### Excluded

- Breadcrumb localization, which is the second delivery stage
- Page titles, forms, tables, notifications, and other page content
- Provider, model, product, and protocol names such as LiteLLM, OpenAI, AI, API, and MCP
- Dark-theme work. The current local branch intentionally keeps the dark-mode control disabled
- Server-side or account-wide preference storage
- New locale-prefixed routes such as `/ru/ui`

## User Experience

On the first authenticated visit, the dashboard reads the browser language. A language beginning with `ru` selects Russian; every other value selects English. If the user changes the language, that explicit choice is saved in local storage and takes precedence on subsequent visits in the same browser.

The language selector appears in the dashboard header near the account and notification controls, where the previous UI placed personal appearance controls. It shows `Русский` and `English`. Selecting either option updates the sidebar and account menu immediately.

The sidebar remains structurally identical in both languages. Routes, permissions, icons, active-item state, collapsed behavior, and nested-menu behavior do not change. Breadcrumbs continue to display their current English section and page names during this stage.

## Localization Architecture

Use `i18next` with `react-i18next` as the reusable localization foundation for later stages. Keep locale resources in small typed modules under the dashboard source tree, with English as the fallback language.

A client-side localization provider owns initialization and exposes the active language. Initialization follows this precedence:

1. A valid saved preference (`en` or `ru`)
2. `navigator.language`, mapping `ru` and `ru-*` to Russian
3. English fallback

The provider does not render the localized dashboard shell until the initial locale is resolved, preventing an English-to-Russian flash and hydration mismatch. Changing the language updates i18next, local storage, and the document `lang` attribute.

Navigation identity remains independent from presentation. Existing page keys, routes, role lists, and the canonical English breadcrumb values remain unchanged. The sidebar rendering layer resolves translated section and item labels from stable navigation keys. This lets stage two localize breadcrumbs deliberately instead of changing them as an accidental side effect.

## Translation Boundaries

Translations use natural Russian UI terminology rather than literal word-for-word output. Technical abbreviations remain recognizable. Examples include:

- `AI Gateway` -> `AI-шлюз`
- `Virtual Keys` -> `Виртуальные ключи`
- `Models + Endpoints` -> `Модели и эндпоинты`
- `MCP Servers` -> `MCP-серверы`
- `Guardrails` -> `Ограничители`
- `Usage` -> `Использование`
- `Access Control` -> `Управление доступом`
- `Developer Tools` -> `Инструменты разработчика`
- `Settings` -> `Настройки`

The complete dictionary covers every sidebar group, every currently reachable top-level and nested item, the sidebar collapse controls, and the account popover. Tests fail if a navigation entry lacks either an English or Russian value.

## Error Handling

- Invalid or unavailable saved language values are ignored
- Missing Russian keys fall back to English
- Local-storage read or write failures do not block the dashboard; the active language remains usable for the current session
- A failed language change leaves the previous language active
- Localization never changes authorization or routing data

## Testing

Use test-driven development for the implementation. Focused tests cover:

- Russian browser locale selects Russian when no preference exists
- Non-Russian browser locale selects English
- A saved valid preference overrides browser detection
- An invalid saved value falls back to browser detection
- Changing the selector updates visible sidebar labels without reload and persists the choice
- All sidebar groups and nested entries have both locale values
- Collapsed tooltips and accessibility labels use the active language
- Account-menu controls use the active language
- Existing role filtering, active item, nested expansion, and route behavior remain unchanged
- `getBreadcrumb` continues returning English values in stage one

Verification includes focused Vitest files, dashboard lint/type checks applicable to the changed files, a production dashboard build, and a visual browser check at desktop and collapsed-sidebar widths.

## Deployment

Build the repository Dockerfile locally for `linux/amd64` and tag the image with an immutable identifier derived from the Git commit. Transfer the compressed image archive to the VPS and load it into Docker. Do not build the full image on the memory-constrained VPS.

Before rollout, record the currently running image and confirm PostgreSQL and Caddy health. Configure the deployment to select the custom image without embedding secrets or replacing data volumes. Recreate only the LiteLLM service, wait for health, then verify:

- the public liveness endpoint
- authenticated UI login
- Russian auto-detection in a Russian browser profile
- manual switching between Russian and English
- persistence after reload
- unchanged English breadcrumbs
- provider configuration and existing database state remain present

If health or UI verification fails, restore the previous image reference and recreate only the LiteLLM service.

## Delivery Checkpoints

1. Locale foundation and tests
2. Russian sidebar and account-menu translations
3. Header selector and local visual verification
4. Custom image build
5. VPS rollout and live verification

Progress is reported after every checkpoint. Implementation and review are performed locally without subagents, as requested.
