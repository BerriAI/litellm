/**
 * Source of truth for the App Router migration smoke (tests/migration/migratedPages.spec.ts).
 *
 * Add a route segment here once its migration has MERGED to the branch under test.
 * Both suites pick it up automatically:
 *   - default mount:           npm run e2e:migration
 *   - server-root-path mount:  SERVER_ROOT_PATH=/<root> npm run e2e:migration:root
 *
 * Keep this in lockstep with MIGRATED_PAGES in src/utils/migratedPages.ts.
 * Pending (add as each PR lands): the leaf-pages batch (budgets, caching,
 * cost-tracking, guardrails, guardrails-monitor, logs, mcp-servers, memory,
 * policies, projects, prompts, search-tools, skills, tag-management,
 * tool-policies, transform-request, ui-theme, vector-stores, workflows,
 * access-groups).
 */
export const MIGRATED_E2E_SEGMENTS: string[] = ["api-reference", "playground"];
