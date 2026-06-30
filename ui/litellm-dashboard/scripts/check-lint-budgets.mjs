import { readFileSync } from "fs";
import { countBudgetViolations, findDrift } from "./lint-budget-lib.mjs";

const argv = process.argv.slice(2);
const positional = [];
const flags = {};
for (let i = 0; i < argv.length; i += 1) {
  if (argv[i] === "--check") {
    flags.check = argv[(i += 1)];
  } else {
    positional.push(argv[i]);
  }
}

const [reportPath, budgetsPath] = positional;
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

if (flags.check) {
  const committed = JSON.parse(readFileSync(flags.check, "utf8"));
  const drift = findDrift(committed, counts);
  for (const { rule, committed: was, actual } of drift) {
    console.error(
      `::error::${flags.check} is stale for ${rule}: committed ${was ?? "missing"}, actual ${actual ?? "not a tracked rule"}.`,
    );
  }
  if (drift.length > 0) {
    console.error(`::error::Run \`npm run lint:metrics\` and commit ${flags.check}.`);
    failed = true;
  } else {
    console.log(`${flags.check} is up to date.`);
  }
}

process.exit(failed ? 1 : 0);
