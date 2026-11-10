/**
 * Providers and upstream gateways report `usage.cost` unvalidated: it arrives as a number, a numeric
 * string, an empty string, or something non-numeric. A cost that does not resolve to a finite number
 * must be dropped rather than coerced, because `NaN` survives `JSON.stringify` as `null` and crashes
 * the metrics row on the next load.
 */
export function parseUsageCost(rawCost: unknown): number | undefined {
  if (typeof rawCost === "number") {
    return Number.isFinite(rawCost) ? rawCost : undefined;
  }

  if (typeof rawCost !== "string") {
    return undefined;
  }

  const trimmed = rawCost.trim();
  if (trimmed === "") {
    return undefined;
  }

  const parsed = Number(trimmed);
  return Number.isFinite(parsed) ? parsed : undefined;
}
