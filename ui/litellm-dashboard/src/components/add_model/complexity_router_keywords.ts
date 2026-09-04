import { KeywordTierRule } from "./KeywordTierRules";

/**
 * Stored shape of a keyword tier rule inside `complexity_router_config`. The UI's
 * KeywordTierRule carries an extra `id` used only as a React key, so it is stripped on the
 * way out and synthesized on the way back in. Both the create form and the edit modal go
 * through here so the two directions cannot drift. The tier is any active tier name: a
 * built-in one, or with tier_definitions, an operator-defined one, so hydration must not
 * filter on the built-in set or an edit would silently delete a custom tier's rules.
 */
export interface StoredKeywordTierRule {
  keywords: string[];
  tier: string;
}

const asKeywords = (value: unknown): string[] =>
  Array.isArray(value)
    ? value.filter((keyword): keyword is string => typeof keyword === "string").map((keyword) => keyword.trim())
    : [];

/**
 * Drop the React-only id and trim keywords, leaving one entry per rule. A rule left empty stays
 * empty rather than disappearing, so getKeywordTierRulesError can name the row it came from.
 */
export const serializeKeywordTierRules = (rules: KeywordTierRule[]): StoredKeywordTierRule[] =>
  rules.map((rule) => ({ keywords: asKeywords(rule.keywords).filter(Boolean), tier: rule.tier }));

/**
 * Positions of the rules left without a keyword, as indexes into the caller's own array. The
 * submit-time message and the inline error on the row both read this, so the row the message
 * names is always the row that lights up.
 */
export const emptyKeywordTierRuleIndexes = (rules: KeywordTierRule[]): number[] =>
  serializeKeywordTierRules(rules).flatMap((rule, index) => (rule.keywords.length === 0 ? [index] : []));

export const hydrateKeywordTierRules = (value: unknown): KeywordTierRule[] => {
  if (!Array.isArray(value)) return [];
  return value.flatMap((entry, index) => {
    if (typeof entry !== "object" || entry === null) return [];
    const record = entry as Record<string, unknown>;
    const keywords = asKeywords(record.keywords).filter(Boolean);
    const tier = record.tier;
    if (keywords.length === 0 || typeof tier !== "string" || !tier.trim()) return [];
    return [{ id: `stored-${index}`, keywords, tier }];
  });
};
