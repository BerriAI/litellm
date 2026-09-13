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
