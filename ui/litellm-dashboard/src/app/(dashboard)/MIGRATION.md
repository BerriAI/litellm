# App Router migration plan

This is the working plan for moving the dashboard off the legacy `?page=` switch and onto path-based App Router routes. Read this before adding a page or touching routing. The conventions for file structure live in [README.md](./README.md)

## Where we are

All pages historically render from one giant `?page=` if/else switch in `src/app/page.tsx`, called the legacy shell. Path-based routes live under `src/app/(dashboard)/<segment>/page.tsx` and are wrapped by `src/app/(dashboard)/layout.tsx`

`MIGRATED_PAGES` in `src/utils/migratedPages.ts` is the single source of truth for which pages are cut over. Adding an entry there routes the sidebar and deep links to the new path and redirects the legacy `?page=` URL. Removing it rolls the page back. As of this writing only `api_ref` -> `/ui/api-reference` is migrated

The data-driven E2E smoke in `e2e_tests/tests/migration/` reads `e2e_tests/fixtures/migratedPages.ts`. Every migrated page gets an entry there so CI exercises it under both the default and server-root-path configurations

## Auth architecture

One engine, one API:

- `AuthProvider` (`src/contexts/AuthContext.tsx`, mounted at the root): fetches uiConfig first so proxy-rooted URLs are correct, then reads the token cookie, decodes the JWT once for the whole app, and applies any custom `auth_header_name`
- `useAuthorized()` (`src/app/(dashboard)/hooks/useAuthorized.ts`): the only auth API pages and hooks should use. It reads the context and layers on policy: the `admin_ui_disabled` check and the redirect to login

Pages get identity from `useAuthorized()` and data from React Query hooks (which call `useAuthorized()` internally for the key). Nothing else should read the cookie or decode the JWT. The legacy shell still calls `useAuth()` directly; that ends in Phase 1, after which `useAuth` stops being exported and the context becomes private plumbing

## Phases

### Phase 0: one auth engine (done, #30049)

`useAuthorized` was rewritten as a thin policy layer over `AuthContext`, ending the era of two parallel auth systems that each decoded the JWT and ran their own redirects. Unreachable route stubs under `(dashboard)/` were removed separately (#30045, following #28891)

### Phase 1: one shell (next)

Move `src/app/page.tsx` into the route group as `(dashboard)/page.tsx` (route groups do not change URLs, so `/ui/` still resolves). Hoist `Navbar`, `SidebarProvider`, `ThemeProvider`, and `DebugWarningBanner` out of it into `(dashboard)/layout.tsx`, wired with real props, and delete the degraded duplicate shell that exists there today. The legacy page shrinks to just the `?page=` switch

This makes the active-page key a render-time derivation from the pathname and search params (no `useState`, no sync effect), turns sidebar items into plain links, and stops the full shell remount when navigating between legacy and migrated pages. It also switches the legacy shell to `useAuthorized()` so `useAuth` can go private

### Phase 2: break the data coupling

The switch component owns `teams`, `keys`, `organizations`, `modelData`, and `userModels` as `useState` and prop-drills them into every arm. Replace these slice by slice with the existing React Query hooks (`hooks/teams/useTeams.ts` pattern, `hooks/common/queryKeysFactory.ts`) so each page pulls its own data. Query caching provides the fetched-once behavior the lifted state was simulating. This is what makes a page extractable, and it retires the deprecated `hooks/useTeams.tsx` along the way

### Phase 3: page-by-page cutover

One small PR per page, using this recipe:

1. Create `(dashboard)/<segment>/page.tsx` as a thin wrapper that pulls data via hooks and renders the view (extract a `<X>View` from the legacy component if one does not exist)
2. Add the entry to `MIGRATED_PAGES`
3. Delete the switch arm and its prop wiring from the legacy page
4. Add the segment to `e2e_tests/fixtures/migratedPages.ts`

Rollback for any page is reverting one PR. Order by extractability: self-contained pages first (ui-theme, cost-tracking, caching, transform-request), prop-tangled ones last (api-keys with its create/prefill deep-link params, models, teams). Playground is nearly free since its `page.tsx` is already the real implementation

Do not create `(dashboard)/<segment>/page.tsx` files ahead of their cutover PR. Unreachable wrappers rot into stubs that compile but render broken pages, which is why the previous batch was deleted

### Phase 4: endgame

When the switch is empty, `(dashboard)/page.tsx` becomes a redirect to the default page. Delete `legacyPageHref`, `legacyKeyForPathname`, and the page-key plumbing, keeping a small `?page=` to path redirect map for old bookmarks. Sidebar active state comes purely from `usePathname()`. Update the README to describe plain file-structure routing, which at that point is true

## Known follow-ups

- Tighten the decoded-JWT fields on `useAuthorized` from the legacy `any` typing; about 25 call sites pass them where `string` is expected and need fixing together
- Remove the dead `showSSOBanner` from the context, the hook, and the typed test mocks
- Migrate the last two importers of the deprecated `hooks/useTeams.tsx` (`templates/key_info_view.tsx`, `UsagePage/.../EntityUsage.tsx`) to the React Query version, then delete it
- `AuthProvider` and `useUIConfig` both fetch uiConfig on cold load; route the provider through the query client so there is one request
- Derive the decoded JWT fields in the provider with `useMemo` instead of eight `useState` slots
