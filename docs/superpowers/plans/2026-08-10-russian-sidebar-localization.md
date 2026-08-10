# Russian Sidebar Localization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add persisted English/Russian localization for the LiteLLM AI Gateway sidebar and account menu, expose the selector to every user in the dashboard header, then deploy the locally built image to the existing VPS.

**Architecture:** A client-side i18next provider resolves a saved `en`/`ru` preference or derives it from `navigator.language`, blocks the dashboard shell until initialization completes, and updates the document language. Navigation keys remain canonical while the sidebar resolves translated display values at render time; breadcrumb data remains English during stage one. The repository Dockerfile produces an immutable `linux/amd64` image that replaces only the running LiteLLM service.

**Tech Stack:** Next.js 16.2.11, React 18.3.1, TypeScript, i18next 26.3.6, react-i18next 17.0.11, Vitest, Testing Library, Docker BuildKit, Docker Compose, Caddy, PostgreSQL

## Global Constraints

- Work without subagents
- Preserve English and add Russian; do not replace English
- Use a saved valid preference before browser detection
- Map `ru` and `ru-*` browser languages to Russian; map all others to English
- Make the selector available to every authenticated role in the dashboard header
- Keep breadcrumbs and page content English in stage one
- Keep LiteLLM, OpenAI, AI, API, MCP, provider names, and model names untranslated
- Preserve routes, permissions, icons, active state, nested navigation, database data, Caddy data, configuration, and certificates
- Build the custom image locally for `linux/amd64`; do not build it on the VPS
- Recreate only the LiteLLM service during rollout and rollback

---

### Task 1: Localization foundation

**Files:**
- Modify: `ui/litellm-dashboard/package.json`
- Modify: `ui/litellm-dashboard/package-lock.json`
- Create: `ui/litellm-dashboard/src/i18n/language.ts`
- Create: `ui/litellm-dashboard/src/i18n/resources.ts`
- Create: `ui/litellm-dashboard/src/i18n/I18nProvider.tsx`
- Create: `ui/litellm-dashboard/src/i18n/language.test.ts`
- Create: `ui/litellm-dashboard/src/i18n/I18nProvider.test.tsx`
- Modify: `ui/litellm-dashboard/src/app/layout.tsx`
- Modify: `ui/litellm-dashboard/src/app/(dashboard)/layout.tsx`
- Modify: `ui/litellm-dashboard/src/app/(dashboard)/layout.test.tsx`

**Interfaces:**
- Produces: `type SupportedLanguage = "en" | "ru"`
- Produces: `const LANGUAGE_STORAGE_KEY = "litellm_ui_language"`
- Produces: `resolveLanguage(saved: string | null, browserLanguage: string): SupportedLanguage`
- Produces: `I18nProvider({ children }: PropsWithChildren): React.ReactNode`
- Produces: `useDashboardLanguage(): { language: SupportedLanguage; setLanguage(language: SupportedLanguage): Promise<void> }`

- [ ] **Step 1: Install exact dependencies**

Run:

```bash
cd ui/litellm-dashboard
npm install --save-exact i18next@26.3.6 react-i18next@17.0.11
```

Expected: `package.json` and `package-lock.json` record the exact versions.

- [ ] **Step 2: Write failing language-resolution tests**

Create table-driven tests that assert:

```ts
expect(resolveLanguage(null, "ru-RU")).toBe("ru");
expect(resolveLanguage(null, "en-US")).toBe("en");
expect(resolveLanguage("en", "ru-RU")).toBe("en");
expect(resolveLanguage("ru", "en-US")).toBe("ru");
expect(resolveLanguage("de", "ru-RU")).toBe("ru");
```

- [ ] **Step 3: Run the focused test and verify RED**

Run: `cd ui/litellm-dashboard && npm test -- --run src/i18n/language.test.ts`

Expected: FAIL because `resolveLanguage` does not exist.

- [ ] **Step 4: Implement language resolution and typed resources**

Implement a pure resolver that accepts only `en` and `ru` from storage, otherwise checks `browserLanguage.toLowerCase().startsWith("ru")`. Define resources with `as const` and English as the fallback language. Initially include the selector labels required by provider tests.

- [ ] **Step 5: Write failing provider tests**

Test initialization from saved preference and browser language, update without reload, local-storage persistence, `document.documentElement.lang`, invalid storage recovery, and graceful local-storage exceptions.

- [ ] **Step 6: Run provider tests and verify RED**

Run: `cd ui/litellm-dashboard && npm test -- --run src/i18n/I18nProvider.test.tsx`

Expected: FAIL because the provider and hook do not exist.

- [ ] **Step 7: Implement the provider and mount it around `DashboardShell`**

Create one i18next instance per provider, initialize it with `fallbackLng: "en"`, set `interpolation.escapeValue` to `false`, and render the existing `LoadingScreen` until resolution completes. Wrap `DashboardShell` inside `I18nProvider` within the dashboard layout so login, chat, and public routes are not changed in stage one.

Extend the root Inter font configuration from `subsets: ["latin"]` to `subsets: ["latin", "cyrillic"]` so English and Russian sidebar labels use the same font metrics.

- [ ] **Step 8: Run focused and layout tests**

Run:

```bash
cd ui/litellm-dashboard
npm test -- --run src/i18n/language.test.ts src/i18n/I18nProvider.test.tsx 'src/app/(dashboard)/layout.test.tsx'
```

Expected: PASS.

- [ ] **Step 9: Commit**

```bash
git add ui/litellm-dashboard/package.json ui/litellm-dashboard/package-lock.json ui/litellm-dashboard/src/i18n ui/litellm-dashboard/src/app/layout.tsx ui/litellm-dashboard/src/app/'(dashboard)'/layout.tsx ui/litellm-dashboard/src/app/'(dashboard)'/layout.test.tsx
git commit -m "feat(ui): add dashboard language foundation"
```

---

### Task 2: Localized sidebar and account menu

**Files:**
- Modify: `ui/litellm-dashboard/src/i18n/resources.ts`
- Modify: `ui/litellm-dashboard/src/components/leftnav.tsx`
- Modify: `ui/litellm-dashboard/src/components/leftnav.test.tsx`
- Modify: `ui/litellm-dashboard/src/components/SidebarAccountMenu/SidebarAccountMenu.tsx`
- Modify: `ui/litellm-dashboard/src/components/SidebarAccountMenu/SidebarAccountMenu.test.tsx`

**Interfaces:**
- Consumes: `useTranslation()` from the task 1 provider instance
- Produces: locale keys under `sidebar.groups`, `sidebar.items`, `sidebar.controls`, and `sidebar.account`
- Preserves: `getBreadcrumb(page): { section: string | null; title: string }` English output

- [ ] **Step 1: Write failing Russian sidebar tests**

Render the sidebar under a Russian provider and assert all visible admin groups and top-level items, including `AI-шлюз`, `Виртуальные ключи`, `Модели и эндпоинты`, `Наблюдаемость`, `Управление доступом`, `Инструменты разработчика`, and `Настройки`. Expand `Инструменты`, `Агентные функции`, `Экспериментальные`, and `Настройки` and assert their Russian child labels.

- [ ] **Step 2: Add translation-completeness and breadcrumb-regression tests**

Iterate stable menu keys and assert both locale dictionaries contain non-empty values. Keep explicit assertions:

```ts
expect(getBreadcrumb("api-keys")).toEqual({ section: "AI Gateway", title: "Virtual Keys" });
expect(getBreadcrumb("logs")).toEqual({ section: "Observability", title: "Logs" });
```

- [ ] **Step 3: Run sidebar tests and verify RED**

Run: `cd ui/litellm-dashboard && npm test -- --run src/components/leftnav.test.tsx`

Expected: FAIL because the sidebar still renders canonical English labels.

- [ ] **Step 4: Add the complete English/Russian sidebar dictionary**

Cover every current group, top-level item, nested item, badge-adjacent label, collapse/expand label, external-link tooltip, and account-menu string. Keep `AI`, `API`, and `MCP` intact in Russian values.

- [ ] **Step 5: Resolve labels during rendering without changing menu identity**

Use stable group and item keys for translation lookups. Keep the canonical English menu labels available to `getBreadcrumb` and page utilities. Localize rendered badge labels without changing global `BetaBadge` or `NewBadge` behavior outside the sidebar.

- [ ] **Step 6: Write and run account-menu RED tests**

Assert Russian values for account tier, role, email, user ID, personal toggles, copy labels, and logout. Run:

`cd ui/litellm-dashboard && npm test -- --run src/components/SidebarAccountMenu/SidebarAccountMenu.test.tsx`

Expected before implementation: FAIL on Russian strings.

- [ ] **Step 7: Localize the account menu and rerun focused tests**

Run:

```bash
cd ui/litellm-dashboard
npm test -- --run src/components/leftnav.test.tsx src/components/SidebarAccountMenu/SidebarAccountMenu.test.tsx
```

Expected: PASS with English and Russian cases.

- [ ] **Step 8: Commit**

```bash
git add ui/litellm-dashboard/src/i18n/resources.ts ui/litellm-dashboard/src/components/leftnav.tsx ui/litellm-dashboard/src/components/leftnav.test.tsx ui/litellm-dashboard/src/components/SidebarAccountMenu
git commit -m "feat(ui): localize dashboard sidebar in Russian"
```

---

### Task 3: Header language selector

**Files:**
- Create: `ui/litellm-dashboard/src/components/LanguageSelector/LanguageSelector.tsx`
- Create: `ui/litellm-dashboard/src/components/LanguageSelector/LanguageSelector.test.tsx`
- Modify: `ui/litellm-dashboard/src/components/DashboardHeader.tsx`
- Modify: `ui/litellm-dashboard/src/components/DashboardHeader.test.tsx`
- Modify: `ui/litellm-dashboard/src/i18n/resources.ts`

**Interfaces:**
- Consumes: `useDashboardLanguage()`
- Produces: `LanguageSelector(): React.ReactNode`

- [ ] **Step 1: Write failing selector tests**

Assert the button exposes the active language, both `Русский` and `English` options are reachable, selecting an option calls `setLanguage`, and the control has a localized accessible label.

- [ ] **Step 2: Run selector tests and verify RED**

Run: `cd ui/litellm-dashboard && npm test -- --run src/components/LanguageSelector/LanguageSelector.test.tsx`

Expected: FAIL because the component does not exist.

- [ ] **Step 3: Implement a compact header dropdown**

Use the existing button/popover primitives and a languages icon. Display `RU` or `EN` in the closed control, show full language names in the menu, and keep the selector independent of user role.

- [ ] **Step 4: Mount the selector before the header toolbar separator**

Do not translate `Docs`, blog, community controls, notifications, or breadcrumb content during this stage.

- [ ] **Step 5: Run selector and header tests**

Run:

```bash
cd ui/litellm-dashboard
npm test -- --run src/components/LanguageSelector/LanguageSelector.test.tsx src/components/DashboardHeader.test.tsx
```

Expected: PASS, including unchanged English breadcrumb assertions.

- [ ] **Step 6: Commit**

```bash
git add ui/litellm-dashboard/src/components/LanguageSelector ui/litellm-dashboard/src/components/DashboardHeader.tsx ui/litellm-dashboard/src/components/DashboardHeader.test.tsx ui/litellm-dashboard/src/i18n/resources.ts
git commit -m "feat(ui): add dashboard language selector"
```

---

### Task 4: Local verification and immutable image

**Files:**
- Verify: `ui/litellm-dashboard/**`
- Verify: `Dockerfile`

**Interfaces:**
- Consumes: the completed localized dashboard
- Produces: Docker image `litellm-russian-sidebar:<commit-sha>` for `linux/amd64`

- [ ] **Step 1: Run the complete affected UI suite**

```bash
cd ui/litellm-dashboard
npm test -- --run src/i18n src/components/leftnav.test.tsx src/components/SidebarAccountMenu/SidebarAccountMenu.test.tsx src/components/LanguageSelector/LanguageSelector.test.tsx src/components/DashboardHeader.test.tsx
npm run lint -- --max-warnings=0
npm run format:check
npm run build
```

Expected: every command exits 0.

- [ ] **Step 2: Run a local browser verification**

Start the dashboard through the project-supported development or container path. Verify Russian auto-detection, English override, persistence after reload, expanded and collapsed sidebar labels, and unchanged English breadcrumbs. Preserve a user-visible browser tab on the verified Russian sidebar.

- [ ] **Step 3: Commit any visual-verification fixes and record the final SHA**

Rerun step 1 after every fix, commit the accepted changes with a focused conventional commit, then run:

`git status --short && git rev-parse HEAD`

Expected: clean feature branch and an immutable final commit.

- [ ] **Step 4: Build the server-architecture image**

```bash
IMAGE_TAG="litellm-russian-sidebar:$(git rev-parse --short=12 HEAD)"
docker buildx build --platform linux/amd64 --load -t "$IMAGE_TAG" .
docker image inspect "$IMAGE_TAG" --format '{{.Architecture}} {{.Id}}'
```

Expected: architecture is `amd64` and the image ID is non-empty.

- [ ] **Step 5: Export and checksum the image**

```bash
docker save "$IMAGE_TAG" | gzip -1 > "/tmp/${IMAGE_TAG/:/-}.tar.gz"
shasum -a 256 "/tmp/${IMAGE_TAG/:/-}.tar.gz" > "/tmp/${IMAGE_TAG/:/-}.tar.gz.sha256"
```

Expected: archive and checksum sidecar exist and are non-empty.

---

### Task 5: Safe VPS rollout

**Files:**
- Modify locally if required: `../ai-gateway-deploy/.env.production`
- Modify on server: `/opt/ai-gateway/.env.production`
- Preserve on server: `/opt/ai-gateway/compose.yml`, `/opt/ai-gateway/config/litellm.yaml`, `/opt/ai-gateway/caddy/Caddyfile`

**Interfaces:**
- Consumes: checked image archive and immutable image tag from task 4
- Produces: healthy public LiteLLM UI at `https://213.193.196.122:8888/ui/`

- [ ] **Step 1: Capture pre-rollout state**

Over SSH, record `docker compose ps`, the resolved LiteLLM image, liveness response, and the current `LITELLM_IMAGE` value without printing secret variables. Confirm PostgreSQL is healthy and Caddy is running.

- [ ] **Step 2: Upload and verify the image archive**

Copy only the archive and checksum sidecar to a dedicated server temporary path. Verify SHA-256 before `docker load`.

- [ ] **Step 3: Load the image and update only `LITELLM_IMAGE`**

Preserve `.env.production` mode `0600`. Replace the exact `LITELLM_IMAGE=` line with the immutable local tag; do not rewrite or print the rest of the file.

- [ ] **Step 4: Recreate only LiteLLM**

```bash
cd /opt/ai-gateway
docker compose --env-file .env.production up -d --no-deps litellm
```

Wait for the LiteLLM health state with a bounded timeout. Do not recreate PostgreSQL or Caddy and do not remove volumes.

- [ ] **Step 5: Verify server and public behavior**

Run the deployment status and smoke commands, verify the public liveness endpoint with the internal CA, confirm the UI login page, and use the browser to validate Russian auto-detection, manual English/Russian switching, persistence, and English breadcrumbs. Confirm existing provider/model rows remain present.

- [ ] **Step 6: Roll back on any failed verification**

Restore the captured image reference, run the same `up -d --no-deps litellm`, and repeat health and smoke checks. Keep the failed image for diagnosis; do not delete data or volumes.

- [ ] **Step 7: Clean explicit temporary archives and report the live image**

Remove only the uploaded archive and checksum sidecar after successful verification. Report the immutable Git SHA, image tag, live URL, and performed checks.
