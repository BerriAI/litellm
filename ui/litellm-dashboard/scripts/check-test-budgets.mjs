import { readFileSync } from "fs";
import { TIER_BUDGET_MS, overBudgetTests, countByTier, tiersOverBudget, budgetGateFails } from "./test-budget-lib.mjs";

const say = (line) => process.stdout.write(`${line}\n`);

const headroomNote = (count, max) => {
  if (count > max) return "OVER BUDGET";
  if (count === max) return "at max";
  return `${max - count} of headroom`;
};

const [budgetsPath, ...reportPaths] = process.argv.slice(2);
const budgets = JSON.parse(readFileSync(budgetsPath, "utf8"));
const report = {
  testResults: reportPaths.flatMap((path) => JSON.parse(readFileSync(path, "utf8")).testResults ?? []),
};
if (report.testResults.length === 0) {
  console.error("::error::no test results were found in the reports handed to the budget gate");
  process.exit(1);
}
say(`Checking ${report.testResults.length} test files from ${reportPaths.length} report(s).`);

const repoRoot = `${process.cwd()}/`;
const violations = overBudgetTests(report, { repoRoot, slackFactor: budgets.slackFactor });
const counts = countByTier(violations);

for (const [tier, baseBudgetMs] of Object.entries(TIER_BUDGET_MS)) {
  const count = counts[tier] ?? 0;
  const max = budgets.maxOverBudget[tier] ?? 0;
  const budgetMs = baseBudgetMs * budgets.slackFactor;
  say(`${tier}: ${count} test(s) over ${budgetMs}ms | max: ${max} | ${headroomNote(count, max)}`);
}

const worst = [...violations].sort((a, b) => b.durationMs - a.durationMs).slice(0, 15);
if (worst.length > 0) {
  say("\nSlowest tests against their tier budget:");
  for (const v of worst) say(`  ${String(v.durationMs).padStart(6)}ms  [${v.tier}] ${v.file} :: ${v.name}`);
}

const failedTiers = tiersOverBudget(counts, budgets);
for (const tier of failedTiers) {
  console.error(
    `::${budgets.enforce ? "error" : "warning"}::${tier} tier has ${counts[tier]} test(s) over ${TIER_BUDGET_MS[tier] * budgets.slackFactor}ms, above the ${budgets.maxOverBudget[tier]} allowed. A multi-second test is what makes CI fail under load. Split it, or move its assertions to a faster tier; lower the max in test-budgets.json as the count drops.`,
  );
}
if (!budgets.enforce) {
  say('\nReporting only: set "enforce": true in test-budgets.json to make these counts a gate.');
}
process.exit(budgetGateFails(counts, budgets) ? 1 : 0);
