export interface TokenBreakdown {
  inputTokens: number | null;
  outputTokens: number | null;
  cacheReadTokens: number | null;
  cacheWriteTokens: number | null;
}

const knownCount = (value: number | null): number | null =>
  value !== null && Number.isFinite(value) && value >= 0 ? value : null;

export const reconciledTokenBreakdown = (total: number, breakdown: TokenBreakdown): TokenBreakdown => {
  const inputTokens = knownCount(breakdown.inputTokens);
  const outputTokens = knownCount(breakdown.outputTokens);
  const cacheReadTokens = knownCount(breakdown.cacheReadTokens);
  const cacheWriteTokens = knownCount(breakdown.cacheWriteTokens);
  const inputReconciles =
    inputTokens !== null && outputTokens !== null && inputTokens + outputTokens === knownCount(total);
  const cacheCountsKnown = cacheReadTokens !== null && cacheWriteTokens !== null;
  const cacheReconciles = inputReconciles && cacheCountsKnown && cacheReadTokens + cacheWriteTokens <= inputTokens;

  return {
    inputTokens,
    outputTokens,
    cacheReadTokens: cacheReconciles ? cacheReadTokens : null,
    cacheWriteTokens: cacheReconciles ? cacheWriteTokens : null,
  };
};
