/**
 * Source of truth for the App Router migration E2E suites.
 *
 * Add an entry (legacy sidebar page id -> route segment) once a page's migration
 * has MERGED to the branch under test. Consumers pick it up automatically:
 *   - migration smoke (tests/migration/migratedPages.spec.ts), via MIGRATED_E2E_SEGMENTS:
 *       default mount:           npm run e2e:migration
 *       server-root-path mount:  SERVER_ROOT_PATH=/<root> npm run e2e:migration:root
 *   - navigation specs that assert per-page URLs (tests/navigation/sidebar.spec.ts)
 *
 * Keep this in lockstep with MIGRATED_PAGES in src/utils/migratedPages.ts.
 * Pending (add as each PR lands): the leaf-pages batch
 * (budgets, caching, cost-tracking, guardrails, guardrails-monitor, logs,
 * mcp-servers, memory, policies, projects, prompts, search-tools, skills,
 * tag-management, tool-policies, transform-request, ui-theme, vector-stores,
 * workflows, access-groups).
 */
export const MIGRATED_E2E_PAGES: Record<string, string> = {
  api_ref: "api-reference",
  "llm-playground": "playground",
};

export const MIGRATED_E2E_SEGMENTS: string[] = [...new Set(Object.values(MIGRATED_E2E_PAGES))];
