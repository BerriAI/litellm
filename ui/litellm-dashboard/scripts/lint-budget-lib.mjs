export function countBudgetViolations(report, budgets) {
  const counts = {};
  for (const file of report) {
    for (const message of file.messages) {
      if (message.ruleId in budgets) {
        counts[message.ruleId] = (counts[message.ruleId] || 0) + 1;
      }
    }
  }
  return Object.fromEntries(
    Object.keys(budgets)
      .sort()
      .map((rule) => [rule, counts[rule] || 0]),
  );
}

export function findDrift(committed, actual) {
  const rules = [...new Set([...Object.keys(actual), ...Object.keys(committed)])].sort();
  return rules
    .filter((rule) => committed[rule] !== actual[rule])
    .map((rule) => ({ rule, committed: committed[rule] ?? null, actual: actual[rule] ?? null }));
}
