# Russian UI Stage 1: Shared Surfaces Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Establish the final modular localization foundation and translate the shared shell, breadcrumbs, authentication, onboarding, and public first-party surfaces in English and Russian.

**Architecture:** Move the existing dashboard-only i18next setup to the root application boundary and compose equal English/Russian namespace dictionaries. Drive Ant Design locale and every first-party language selector from the same persisted language context. Migrate only Stage 1 surfaces in this plan; later product-area plans consume the namespaces introduced here.

**Tech Stack:** Next.js 16, React 19, TypeScript, i18next 26, react-i18next 17, Ant Design 6, Vitest, Testing Library, Docker Buildx.

## Global Constraints

- Supported languages are exactly English (`en`) and Russian (`ru`).
- Saved `localStorage` choice wins; otherwise `ru`/`ru-*` browser locales select Russian and every other locale selects English.
- Technical names, provider/model names, API routes, configuration keys, identifiers, code, request/response bodies, and backend-owned messages remain unchanged.
- Missing Russian keys fall back to English; no included surface may render blank copy.
- The external Agent Control Plane is out of scope.
- Work inline without subagents, as explicitly requested by the user.
- Every task follows RED → GREEN → focused regression tests → commit.

---

### Task 1: Modular Translation Catalog

**Files:**
- Create: `ui/litellm-dashboard/src/i18n/catalog.ts`
- Create: `ui/litellm-dashboard/src/i18n/catalog.test.ts`
- Create: `ui/litellm-dashboard/src/i18n/locales/en/common.ts`
- Create: `ui/litellm-dashboard/src/i18n/locales/en/auth.ts`
- Create: `ui/litellm-dashboard/src/i18n/locales/en/navigation.ts`
- Create: `ui/litellm-dashboard/src/i18n/locales/ru/common.ts`
- Create: `ui/litellm-dashboard/src/i18n/locales/ru/auth.ts`
- Create: `ui/litellm-dashboard/src/i18n/locales/ru/navigation.ts`
- Modify: `ui/litellm-dashboard/src/i18n/resources.ts`
- Modify: `ui/litellm-dashboard/src/i18n/sidebar.ts`
- Modify: `ui/litellm-dashboard/src/components/leftnav.tsx`
- Modify: `ui/litellm-dashboard/src/components/SidebarAccountMenu/SidebarAccountMenu.tsx`
- Test: `ui/litellm-dashboard/src/components/leftnav.test.tsx`
- Test: `ui/litellm-dashboard/src/components/SidebarAccountMenu/SidebarAccountMenu.test.tsx`

**Interfaces:**
- Produces: `TRANSLATION_NAMESPACES`, `resources`, and `TranslationNamespace` from `i18n/catalog.ts`.
- Produces: equal `common`, `auth`, and `navigation` key trees for `en` and `ru`.
- Preserves: `getSidebarTranslations(language: SupportedLanguage)` for sidebar consumers.

- [ ] **Step 1: Write the catalog parity test**

  Assert recursively that every Russian namespace has exactly the English leaf paths and that required namespaces are registered:

  ```ts
  expect(TRANSLATION_NAMESPACES).toEqual(["common", "auth", "navigation"]);
  expect(leafPaths(resources.ru.navigation)).toEqual(leafPaths(resources.en.navigation));
  expect(leafPaths(resources.ru.auth)).toEqual(leafPaths(resources.en.auth));
  ```

- [ ] **Step 2: Run the test and verify RED**

  Run in the pinned Node container:

  ```bash
  npm test -- src/i18n/catalog.test.ts
  ```

  Expected: FAIL because `catalog.ts` and locale modules do not exist.

- [ ] **Step 3: Create the namespace modules and catalog**

  Export focused plain objects and compose them without duplicating keys:

  ```ts
  export const TRANSLATION_NAMESPACES = ["common", "auth", "navigation"] as const;
  export type TranslationNamespace = (typeof TRANSLATION_NAMESPACES)[number];

  export const resources = {
    en: { common: enCommon, auth: enAuth, navigation: enNavigation },
    ru: { common: ruCommon, auth: ruAuth, navigation: ruNavigation },
  } as const;
  ```

  Move the existing language and sidebar/account dictionaries into `common` and `navigation`. Add Stage 1 keys for common actions, loading/empty states, breadcrumbs, header controls, login, onboarding, and public pages. Keep `resources.ts` as a compatibility re-export during this stage:

  ```ts
  export { resources } from "./catalog";
  ```

- [ ] **Step 4: Migrate existing sidebar consumers**

  Update direct paths from `resources[language].translation.sidebar` to `resources[language].navigation.sidebar`; retain canonical English page identifiers for routing and interpolate only visible labels.

- [ ] **Step 5: Run catalog and existing navigation tests**

  ```bash
  npm test -- src/i18n/catalog.test.ts src/components/leftnav.test.tsx src/components/SidebarAccountMenu/SidebarAccountMenu.test.tsx
  ```

  Expected: all tests PASS in English and Russian.

- [ ] **Step 6: Commit**

  ```bash
  git commit -m "refactor(ui): modularize localization catalog"
  ```

### Task 2: Root Localization and Ant Design Locale

**Files:**
- Modify: `ui/litellm-dashboard/src/i18n/I18nProvider.tsx`
- Modify: `ui/litellm-dashboard/src/i18n/I18nProvider.test.tsx`
- Modify: `ui/litellm-dashboard/src/app/layout.tsx`
- Create: `ui/litellm-dashboard/src/app/layout.test.tsx`
- Modify: `ui/litellm-dashboard/src/app/(dashboard)/layout.tsx`
- Modify: `ui/litellm-dashboard/src/app/(dashboard)/layout.test.tsx`
- Modify: `ui/litellm-dashboard/src/contexts/AntdGlobalProvider.tsx`
- Create: `ui/litellm-dashboard/src/contexts/AntdGlobalProvider.test.tsx`

**Interfaces:**
- Consumes: `resources` and `TRANSLATION_NAMESPACES` from Task 1.
- Produces: one root `I18nProvider` for dashboard, login, onboarding, public pages, connect, and Chat routes.
- Preserves: `useDashboardLanguage(): { language; setLanguage }` until the full localization program can rename the public hook safely.

- [ ] **Step 1: Write failing provider-boundary tests**

  Assert that RootLayout wraps `AntdGlobalProvider` and all children with `I18nProvider`, the dashboard layout no longer nests a second instance, and Ant Design receives the matching locale:

  ```ts
  expect(mockConfigProvider).toHaveBeenCalledWith(expect.objectContaining({ locale: ruRU }), expect.anything());
  expect(document.documentElement.lang).toBe("ru");
  ```

- [ ] **Step 2: Run the focused tests and verify RED**

  ```bash
  npm test -- src/app/layout.test.tsx src/contexts/AntdGlobalProvider.test.tsx src/i18n/I18nProvider.test.tsx
  ```

  Expected: FAIL because localization is still dashboard-scoped and ConfigProvider has no locale.

- [ ] **Step 3: Initialize i18next with explicit namespaces**

  Use these options in `I18nProvider`:

  ```ts
  const initializationOptions = {
    resources,
    ns: TRANSLATION_NAMESPACES,
    defaultNS: "common",
    fallbackLng: "en",
    fallbackNS: "common",
    lng: initialLanguage,
    interpolation: { escapeValue: false },
  };
  ```

- [ ] **Step 4: Move the provider to RootLayout**

  Wrap `AntdGlobalProvider` with `I18nProvider` in `app/layout.tsx`, remove only the redundant dashboard wrapper, and keep all authentication and React Query boundaries otherwise unchanged.

- [ ] **Step 5: Synchronize Ant Design locale**

  Import `enUS` and `ruRU` from `antd/locale`, read `language` from the shared context, and pass `locale={language === "ru" ? ruRU : enUS}` to ConfigProvider.

- [ ] **Step 6: Run provider, layout, and existing affected tests**

  ```bash
  npm test -- src/app/layout.test.tsx src/contexts/AntdGlobalProvider.test.tsx src/i18n/I18nProvider.test.tsx "src/app/(dashboard)/layout.test.tsx"
  ```

  Expected: PASS with one language instance across route groups.

- [ ] **Step 7: Commit**

  ```bash
  git commit -m "feat(ui): provide localization across all routes"
  ```

### Task 3: Shared Navigation, Breadcrumbs, and Language Access

**Files:**
- Modify: `ui/litellm-dashboard/src/components/DashboardHeader.tsx`
- Modify: `ui/litellm-dashboard/src/components/DashboardHeader.test.tsx`
- Modify: `ui/litellm-dashboard/src/components/Navbar/ViewSwitcher.tsx`
- Modify: `ui/litellm-dashboard/src/components/Navbar/ViewSwitcher.test.tsx`
- Modify: `ui/litellm-dashboard/src/components/navbar.tsx`
- Modify: `ui/litellm-dashboard/src/components/navbar.test.tsx`
- Modify: `ui/litellm-dashboard/src/components/page_metadata.ts`
- Modify: `ui/litellm-dashboard/src/components/LanguageSelector/LanguageSelector.tsx`
- Modify: `ui/litellm-dashboard/src/components/LanguageSelector/LanguageSelector.test.tsx`

**Interfaces:**
- Consumes: `navigation` and `common` namespaces.
- Produces: localized `DashboardHeader`, legacy Navbar controls, page titles, breadcrumbs, and view-switcher copy.
- Preserves: route keys and plugin display names exactly as received.

- [ ] **Step 1: Add failing English/Russian navigation tests**

  Cover breadcrumb titles, `Docs`, Chat availability text, selector accessible name, legacy Navbar actions, and plugin-name preservation:

  ```ts
  expect(screen.getByText("Ограничители")).toBeInTheDocument();
  expect(screen.getByRole("button", { name: "Язык: Русский" })).toBeVisible();
  expect(screen.getByText("Agent Control Plane")).toBeInTheDocument();
  ```

- [ ] **Step 2: Run tests and verify RED**

  ```bash
  npm test -- src/components/DashboardHeader.test.tsx src/components/Navbar/ViewSwitcher.test.tsx src/components/navbar.test.tsx src/components/LanguageSelector/LanguageSelector.test.tsx
  ```

- [ ] **Step 3: Translate visible navigation through keys**

  Use `useTranslation(["navigation", "common"])`. Map canonical page IDs to `navigation:breadcrumbs.<page>` and use English fallback for unknown plugin-provided routes. Translate the disabled Chat explanation and shared Navbar actions; do not translate plugin display names.

- [ ] **Step 4: Keep the selector recognizable on all widths**

  Retain the globe icon and language code, add a visible localized tooltip/title, and preserve the accessible label. Do not move the selector out of the shared header in this task.

- [ ] **Step 5: Run navigation tests**

  Expected: all focused tests PASS for both locales.

- [ ] **Step 6: Commit**

  ```bash
  git commit -m "feat(ui): localize shared navigation and breadcrumbs"
  ```

### Task 4: Login and Onboarding

**Files:**
- Modify: `ui/litellm-dashboard/src/app/login/LoginPage.tsx`
- Create: `ui/litellm-dashboard/src/app/login/LoginPage.test.tsx`
- Modify: `ui/litellm-dashboard/src/app/onboarding/page.tsx`
- Modify: `ui/litellm-dashboard/src/app/onboarding/OnboardingForm.tsx`
- Create: `ui/litellm-dashboard/src/app/onboarding/OnboardingForm.test.tsx`
- Modify: `ui/litellm-dashboard/src/components/LanguageSelector/LanguageSelector.tsx`

**Interfaces:**
- Consumes: `auth` and `common` namespaces and the root provider from Task 2.
- Produces: localized login, disabled-UI state, worker selection, signup, invitation, and password-reset flows.
- Preserves: raw login mutation errors and technical values such as `MASTER_KEY` and `DISABLE_ADMIN_UI=False`.

- [ ] **Step 1: Write failing auth-flow tests**

  Cover Russian login labels and validation, English switching, admin-disabled copy, worker selection, onboarding variants, and language selector visibility:

  ```ts
  expect(screen.getByRole("heading", { name: "Вход" })).toBeInTheDocument();
  expect(screen.getByLabelText("Имя пользователя")).toBeEnabled();
  expect(screen.getByText("MASTER_KEY")).toBeInTheDocument();
  ```

- [ ] **Step 2: Run auth tests and verify RED**

  ```bash
  npm test -- src/app/login/LoginPage.test.tsx src/app/onboarding/OnboardingForm.test.tsx
  ```

- [ ] **Step 3: Replace frontend-owned auth copy with translation keys**

  Translate headings, descriptions, hints, form labels, placeholders, validation, buttons, disabled UI text, signup/reset states, and loading fallback. Render dynamic worker names and backend errors unchanged.

- [ ] **Step 4: Add the selector to auth surfaces**

  Place the reusable selector in a fixed top-right auth toolbar with an explicit background and focus style so it remains visible independently of the card width.

- [ ] **Step 5: Run auth and provider regression tests**

  Expected: all tests PASS and switching changes the visible auth form without reload.

- [ ] **Step 6: Commit**

  ```bash
  git commit -m "feat(ui): localize login and onboarding"
  ```

### Task 5: Public First-Party Surfaces

**Files:**
- Modify: `ui/litellm-dashboard/src/components/public_model_hub.tsx`
- Modify: `ui/litellm-dashboard/src/components/public_model_hub.test.tsx`
- Modify: `ui/litellm-dashboard/src/app/model_hub/page.tsx`
- Modify: `ui/litellm-dashboard/src/app/model_hub_table/page.tsx`
- Modify: `ui/litellm-dashboard/src/app/mcp/oauth/callback/page.tsx`
- Create: `ui/litellm-dashboard/src/app/mcp/oauth/callback/page.test.tsx`

**Interfaces:**
- Consumes: `common`, `auth`, and `navigation` namespaces.
- Produces: localized public model hub/table shell and OAuth callback frontend states.
- Preserves: model/provider names, endpoint modes, API examples, and backend-returned descriptions.

- [ ] **Step 1: Write failing public-surface tests**

  Assert Russian headings, searches, filters, empty/loading/error states, action labels, OAuth completion copy, English switching, and unchanged model/provider values.

- [ ] **Step 2: Run focused tests and verify RED**

  ```bash
  npm test -- src/components/public_model_hub.test.tsx src/app/mcp/oauth/callback/page.test.tsx
  ```

- [ ] **Step 3: Translate only frontend-owned public copy**

  Use namespace keys for navigation, explanations, filters, buttons, tooltips, and statuses. Interpolate counts and names; keep backend catalog data intact.

- [ ] **Step 4: Make the selector available on public layouts**

  Reuse the auth/public top-right selector placement without introducing a second provider.

- [ ] **Step 5: Run public and full Stage 1 affected tests**

  Expected: focused public tests and all tests from Tasks 1–4 PASS.

- [ ] **Step 6: Commit**

  ```bash
  git commit -m "feat(ui): localize public LiteLLM surfaces"
  ```

### Task 6: Stage 1 Completeness Audit and Production Verification

**Files:**
- Create: `ui/litellm-dashboard/scripts/audit-localization.mjs`
- Create: `ui/litellm-dashboard/scripts/audit-localization.test.ts`
- Modify: `ui/litellm-dashboard/package.json`
- Modify: `ui/litellm-dashboard/package-lock.json`

**Interfaces:**
- Produces: `npm run i18n:audit` with a path-scoped allowlist for technical literals.
- Consumes: all Stage 1 migrated file paths and the equal namespace catalog.

- [ ] **Step 1: Write a failing audit fixture test**

  Prove the audit reports raw JSX copy while allowing technical literals:

  ```ts
  expect(audit('return <Button>Save changes</Button>')).toContain("Save changes");
  expect(audit('return <code>MASTER_KEY</code>')).toEqual([]);
  ```

- [ ] **Step 2: Run the test and verify RED**

  ```bash
  npm test -- scripts/audit-localization.test.ts
  ```

- [ ] **Step 3: Implement the scoped audit command**

  Scan only the Stage 1 source files for user-visible raw JSX text and accessibility attributes. Keep every exception in a named allowlist containing a reason; fail on undocumented additions.

- [ ] **Step 4: Run Stage 1 verification**

  In the pinned Node 24 container run:

  ```bash
  npm run i18n:audit
  npm run format:check
  npm run lint -- --quiet
  npm test -- src/i18n src/app/layout.test.tsx src/contexts/AntdGlobalProvider.test.tsx src/components/DashboardHeader.test.tsx src/components/Navbar/ViewSwitcher.test.tsx src/components/navbar.test.tsx src/components/LanguageSelector/LanguageSelector.test.tsx src/components/leftnav.test.tsx src/components/SidebarAccountMenu/SidebarAccountMenu.test.tsx src/app/login/LoginPage.test.tsx src/app/onboarding/OnboardingForm.test.tsx src/components/public_model_hub.test.tsx src/app/mcp/oauth/callback/page.test.tsx
  npm run build
  ```

  Expected: audit, formatting, error-only lint, all affected tests, and production build PASS. Existing unrelated warning debt may be reported separately.

- [ ] **Step 5: Commit**

  ```bash
  git commit -m "test(ui): audit shared localization coverage"
  ```

### Task 7: Build and Deploy Stage 1

**Files:**
- Modify (ignored local state): `/Users/aleksander/Documents/MyApps/liga-apps/ai-gateway-deploy/.env.local`
- Modify (ignored local state): `/Users/aleksander/Documents/MyApps/liga-apps/ai-gateway-deploy/.env.production`
- Modify remotely: `/opt/ai-gateway/.env.production`

**Interfaces:**
- Consumes: the verified final Stage 1 commit.
- Produces: immutable image `litellm-russian-stage1:${STAGE1_SHA}` for `linux/amd64`, where `STAGE1_SHA` is the verified commit's 12-character SHA.
- Preserves: PostgreSQL, Caddy, volumes, config, secrets, certificates, and the previous image.

- [ ] **Step 1: Build and inspect the immutable image**

  ```bash
  STAGE1_SHA="$(git rev-parse --short=12 HEAD)"
  STAGE1_IMAGE="litellm-russian-stage1:${STAGE1_SHA}"
  docker buildx build --platform linux/amd64 --load -t "${STAGE1_IMAGE}" .
  docker image inspect "${STAGE1_IMAGE}"
  ```

  Verify `linux/amd64`, LiteLLM `1.97.0`, and Russian Stage 1 strings in packaged JS.

- [ ] **Step 2: Back up production**

  Create a timestamped PostgreSQL custom-format dump and a `0600` archive of `.env.production`, `compose.yml`, Caddyfile, LiteLLM config, and CA certificate. Verify non-empty files and SHA-256 checksums.

- [ ] **Step 3: Transfer, verify, and load the image**

  Stream-compress `docker save`, transfer through the authenticated SSH control connection, compare SHA-256 locally/remotely, then `docker load` on the VPS.

- [ ] **Step 4: Recreate only LiteLLM**

  Update all three environment files to the immutable tag and run:

  ```bash
  docker compose --env-file .env.production up -d --no-deps --force-recreate litellm
  ```

- [ ] **Step 5: Verify live service and UI**

  Require `healthy`, zero restarts, PostgreSQL accepting connections, LiteLLM version `1.97.0`, and HTTP 200 for public liveliness, readiness, and `/ui/`. Inspect served JS for Russian breadcrumbs/auth strings and verify the language selector in the live browser.

- [ ] **Step 6: Clean transport artifacts and retain recovery assets**

  Delete only the transferred/local image archives. Keep the previous Docker image and verified database/config backups until the user accepts Stage 1.
