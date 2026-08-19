export const TIER_BUDGET_MS = Object.freeze({ unit: 50, component: 1000, integration: 3000 });

const TEST_TS_PATHS_THAT_RENDER_REACT = Object.freeze([
  /(^|\/)hooks\/.*\.test\.ts$/,
  /cost-tracking\/_components\/(?:.*\/)?use_[^/]*\.test\.ts$/,
  /models-and-endpoints\/detailNavigation\.test\.ts$/,
  /models-and-endpoints\/vertexCredentialsUpload\.test\.ts$/,
  /^src\/components\/chat\/useChatHistory\.test\.ts$/,
  /^src\/lib\/forms\/pickDirty\.test\.ts$/,
]);

export const tierForTestFile = (relativePath) => {
  if (relativePath.includes(".integration.test.")) return "integration";
  if (relativePath.endsWith(".test.tsx")) return "component";
  if (TEST_TS_PATHS_THAT_RENDER_REACT.some((m) => m.test(relativePath))) return "component";
  return "unit";
};

export const overBudgetTests = (report, { repoRoot, slackFactor = 1 }) =>
  report.testResults.flatMap((file) => {
    const relativePath = file.name.startsWith(repoRoot) ? file.name.slice(repoRoot.length) : file.name;
    const tier = tierForTestFile(relativePath);
    const budgetMs = TIER_BUDGET_MS[tier] * slackFactor;
    return (file.assertionResults ?? [])
      .filter((test) => (test.duration ?? 0) > budgetMs)
      .map((test) => ({
        tier,
        budgetMs,
        file: relativePath,
        name: test.fullName ?? test.title,
        durationMs: Math.round(test.duration ?? 0),
      }));
  });

export const countByTier = (violations) =>
  violations.reduce((counts, v) => ({ ...counts, [v.tier]: (counts[v.tier] ?? 0) + 1 }), {});
