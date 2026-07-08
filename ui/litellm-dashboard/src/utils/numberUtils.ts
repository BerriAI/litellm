/**
 * Shared numeric coercion utility.
 *
 * Cost values (e.g. MCP server `default_cost_per_query` and per-tool overrides)
 * are typed as `number | null` in the UI, but values arriving from the backend
 * — JSONB columns, YAML/JSON config import paths, or `antd`'s `InputNumber`
 * which can emit `string` for some precision/locale combinations — may be
 * stringified numbers. Calling numeric methods like `.toFixed(...)` on those
 * crashes the page.
 *
 * Use `toFiniteNumber` to coerce defensively before formatting.
 */

/**
 * Coerce an unknown value to a finite number, or `null` if it cannot be.
 *
 * - `number`: returns the value if finite (rejects `NaN` / `±Infinity`).
 * - `string`: trimmed and parsed via `Number(...)` if non-empty; rejects
 *   results that aren't finite (e.g. empty, whitespace, `"abc"`).
 * - everything else (`null`, `undefined`, objects, booleans): `null`.
 */
export const toFiniteNumber = (value: unknown): number | null => {
  if (typeof value === "number") {
    return Number.isFinite(value) ? value : null;
  }
  if (typeof value === "string") {
    const trimmed = value.trim();
    if (trimmed === "") return null;
    const parsed = Number(trimmed);
    return Number.isFinite(parsed) ? parsed : null;
  }
  return null;
};
