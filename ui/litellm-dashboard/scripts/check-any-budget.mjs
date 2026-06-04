import { readFileSync } from "fs";

const RULE = "@typescript-eslint/no-explicit-any";
const [, , reportPath, budgetPath] = process.argv;

const report = JSON.parse(readFileSync(reportPath, "utf8"));
const { max, target } = JSON.parse(readFileSync(budgetPath, "utf8"));

let count = 0;
for (const file of report) {
  for (const message of file.messages) {
    if (message.ruleId === RULE) count++;
  }
}

console.log(`explicit any: ${count} | budget max: ${max} | target: ${target}`);

if (count > max) {
  console.error(
    `::error::explicit-any budget exceeded (${count} > ${max}). Replace an any with a real type, or you have introduced more than the allowed headroom.`,
  );
  process.exit(1);
}

if (count <= target) {
  console.log(`Target reached. Lower "target" in eslint-any-budget.json to keep ratcheting down.`);
} else if (count < max) {
  console.log(`${max - count} of headroom left; lower "max" toward ${target} as anys are removed.`);
}
