# Complete Russian UI Localization Design

## Goal

Add complete Russian localization to every first-party LiteLLM interface used in
this deployment while retaining English as a fully supported language. Deliver
the work in visible, independently deployable stages so each translated area can
be reviewed in the live environment before the next area is migrated.

## Scope

The localization covers:

- the shared application shell, headers, breadcrumbs, account controls, login,
  onboarding, and public first-party pages;
- every page reachable from the AI Gateway sidebar for every supported role;
- modals, forms, validation copy, empty states, tables, filters, tooltips,
  notifications, pagination, and accessibility labels owned by the frontend;
- the built-in Chat UI and its settings, conversation controls, empty states,
  and errors owned by the frontend;
- Ant Design built-in copy, date formatting, and pagination copy where locale
  support is available.

The external Agent Control Plane is excluded. LiteLLM embeds registered plugins
in an iframe, so their UI must be localized in each plugin's own repository.

Technical names remain recognizable and are not translated: LiteLLM, AI, API,
MCP, provider and model names, API routes, configuration keys, identifiers,
code samples, request and response bodies, and messages returned dynamically by
the backend or third-party providers.

## Language Behavior

- Supported languages are English (`en`) and Russian (`ru`).
- A saved user choice in `localStorage` has highest priority.
- On first use, browser locales beginning with `ru` select Russian; all other
  browser locales select English.
- A single persisted choice applies to AI Gateway, login and public pages, and
  Chat UI.
- The language selector is available on every first-party surface: the AI
  Gateway header, Chat UI header, login, onboarding, and public pages.
- Changing language updates the visible interface immediately and updates the
  document `lang` attribute.
- Missing Russian keys fall back to English rather than rendering blank copy.
  Automated checks report missing keys before deployment.

## Architecture

Keep i18next and react-i18next as the localization runtime, but replace the
growing monolithic resource object with bounded dictionaries organized by
product area:

- `common`: shared actions, states, validation, dates, tables, and accessibility;
- `auth`: login, onboarding, invitations, and public authentication copy;
- `navigation`: headers, breadcrumbs, account controls, and navigation;
- `gateway`: keys, Playground, models, agents, MCP, skills, guardrails, policies,
  and tools;
- `observability`: usage, spend, logs, cost optimization, and monitoring;
- `access`: teams, users, organizations, access groups, and budgets;
- `settings`: developer tools, router, alerts, admin settings, cost tracking,
  caching, experimental pages, and UI theme;
- `chat`: the complete built-in Chat UI.

English and Russian expose identical key trees. Components request keys from
their product namespace through react-i18next. Dynamic domain data is passed as
interpolation values and never copied into translation dictionaries.

The localization provider moves to a shared boundary that covers all
first-party routes. Ant Design locale and date locale derive from the same
selected language so component-library copy cannot drift from application copy.

## Delivery Stages

1. **Foundation and shared surfaces**: split dictionaries into namespaces,
   establish the shared provider, localize common components, breadcrumbs,
   headers, login, onboarding, and public first-party pages.
2. **AI Gateway**: virtual keys, Playground, models and endpoints, agents, MCP,
   skills, guardrails, policies, search tools, vector stores, and tool policies.
3. **Observability**: usage, cost optimization, logs, and guardrail monitoring.
4. **Access management**: teams, internal users, organizations, access groups,
   and budgets.
5. **Developer tools and settings**: API reference, AI Hub, caching,
   experimental pages, router settings, logging and alerts, admin settings,
   cost tracking, and UI theme.
6. **Chat UI**: navigation, conversations, integrations, credentials, keys,
   logs, usage, prompts, empty states, and frontend-owned errors.
7. **Completeness audit**: scan all included surfaces, resolve remaining
   frontend-owned English copy, and document intentional technical exceptions.

Each stage is committed, verified, built into a new immutable `linux/amd64`
image, deployed by recreating only the LiteLLM service, and checked in the live
UI before the next stage begins.

## Verification

Each stage must pass:

- dictionary parity tests proving English and Russian contain the same keys;
- focused component tests covering both languages and immediate switching;
- tests that technical data and identifiers remain unchanged;
- an untranslated-copy audit scoped to migrated files with an explicit,
  reviewable allowlist for technical text and backend-owned messages;
- ESLint with no new errors, formatting checks, and the affected Vitest suite;
- a clean Next.js production build;
- inspection of the built image for the expected version and Russian assets;
- live HTTPS checks for liveliness, readiness, UI assets, and the translated
  controls introduced in that stage.

Existing repository-wide warning debt is reported separately and does not hide
new errors in changed files.

## Deployment and Recovery

Before each production replacement, record the running image and verify the
existing database and configuration backup. Preserve PostgreSQL, Caddy,
configuration, secrets, certificates, and volumes. Load the new immutable image,
update `LITELLM_IMAGE` in both local environment files and production, recreate
only LiteLLM, wait for health, and run public smoke checks. Retain the previous
image and database backup until the stage has been visually accepted.

## Acceptance Criteria

The program is complete when:

- every included first-party surface can be used in Russian without unintended
  English UI copy;
- switching to English restores a complete English interface on the same
  surfaces;
- language selection persists across reloads and navigation between AI Gateway,
  public routes, and Chat UI;
- all roles see localized navigation and controls available to them;
- technical names and backend data remain unchanged;
- the completeness audit, tests, production build, image checks, and live smoke
  checks all pass;
- the external Agent Control Plane remains explicitly outside this repository's
  localization boundary.
