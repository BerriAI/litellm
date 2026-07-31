import { ComplexityTier, KeywordTierRule } from "./KeywordTierRules";

/**
 * Stored shape of a keyword tier rule inside `complexity_router_config`. The UI's
 * KeywordTierRule carries an extra `id` used only as a React key, so it is stripped on the
 * way out and synthesized on the way back in. Both the create form and the edit modal go
 * through here so the two directions cannot drift.
 */
export interface StoredKeywordTierRule {
  keywords: string[];
  tier: ComplexityTier;
}

const TIERS: ReadonlySet<string> = new Set<ComplexityTier>(["SIMPLE", "MEDIUM", "COMPLEX", "REASONING"]);

const asKeywords = (value: unknown): string[] =>
  Array.isArray(value)
    ? value.filter((keyword): keyword is string => typeof keyword === "string").map((keyword) => keyword.trim())
    : [];

/**
 * Drop the React-only id, trim keywords, and discard rules left empty. "Add keyword rule"
 * seeds a row with no keywords, and the backend validator rejects those with a 400.
 */
export const serializeKeywordTierRules = (rules: KeywordTierRule[]): StoredKeywordTierRule[] =>
  rules
    .map((rule) => ({ keywords: asKeywords(rule.keywords).filter(Boolean), tier: rule.tier }))
    .filter((rule) => rule.keywords.length > 0);

export const hydrateKeywordTierRules = (value: unknown): KeywordTierRule[] => {
  if (!Array.isArray(value)) return [];
  return value.flatMap((entry, index) => {
    if (typeof entry !== "object" || entry === null) return [];
    const record = entry as Record<string, unknown>;
    const keywords = asKeywords(record.keywords).filter(Boolean);
    const tier = record.tier;
    if (keywords.length === 0 || typeof tier !== "string" || !TIERS.has(tier)) return [];
    return [{ id: `stored-${index}`, keywords, tier: tier as ComplexityTier }];
  });
};
