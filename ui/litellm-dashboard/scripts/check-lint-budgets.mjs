import { readFileSync } from "fs";
import { countBudgetViolations } from "./lint-budget-lib.mjs";

const [reportPath, budgetsPath] = process.argv.slice(2);
const report = JSON.parse(readFileSync(reportPath, "utf8"));
const budgets = JSON.parse(readFileSync(budgetsPath, "utf8"));
const counts = countBudgetViolations(report, budgets);

let failed = false;
for (const [rule, { max, target }] of Object.entries(budgets)) {
  const count = counts[rule];
  const note = count > max ? "OVER BUDGET" : count <= target ? "at target" : `${max - count} of headroom`;
  console.log(`${rule}: ${count} | max: ${max} | target: ${target} | ${note}`);
  if (count > max) {
    console.error(
      `::error::${rule} budget exceeded (${count} > ${max}). Reduce usage; lower max in eslint-budgets.json as the count drops.`,
    );
    failed = true;
  }
}

process.exit(failed ? 1 : 0);
