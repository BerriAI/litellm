import { execSync } from "child_process";
import { mkdtempSync, readFileSync, writeFileSync, rmSync } from "fs";
import { tmpdir } from "os";
import { join } from "path";
import { countBudgetViolations } from "./lint-budget-lib.mjs";

const budgets = JSON.parse(readFileSync("eslint-budgets.json", "utf8"));
const dir = mkdtempSync(join(tmpdir(), "litellm-lint-"));
const reportPath = join(dir, "report.json");

try {
  // eslint exits non-zero whenever any error-level rule fires; the JSON report is still written.
  execSync(`npx eslint . -f json -o "${reportPath}"`, { stdio: "inherit" });
} catch {
  /* report file is what we read below */
}

const report = JSON.parse(readFileSync(reportPath, "utf8"));
rmSync(dir, { recursive: true, force: true });

const metrics = countBudgetViolations(report, budgets);
writeFileSync("eslint-metrics.json", JSON.stringify(metrics, null, 2) + "\n");
console.log("Updated eslint-metrics.json");
console.table(metrics);
