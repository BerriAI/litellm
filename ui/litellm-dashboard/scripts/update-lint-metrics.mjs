import { execSync } from "child_process";
import { mkdtempSync, readFileSync, writeFileSync, rmSync } from "fs";
import { tmpdir } from "os";
import { join } from "path";
import { countBudgetViolations } from "./lint-budget-lib.mjs";

const ESLINT_EXIT_LINT_ERRORS = 1;

const budgets = JSON.parse(readFileSync("eslint-budgets.json", "utf8"));
const dir = mkdtempSync(join(tmpdir(), "litellm-lint-"));
const reportPath = join(dir, "report.json");

try {
  execSync(`npx eslint . -f json -o "${reportPath}"`, { stdio: "inherit" });
} catch (err) {
  if (err.status !== ESLINT_EXIT_LINT_ERRORS) throw err;
}

const report = JSON.parse(readFileSync(reportPath, "utf8"));
rmSync(dir, { recursive: true, force: true });

const metrics = countBudgetViolations(report, budgets);
writeFileSync("eslint-metrics.json", JSON.stringify(metrics, null, 2) + "\n");
console.log("Updated eslint-metrics.json");
console.table(metrics);
