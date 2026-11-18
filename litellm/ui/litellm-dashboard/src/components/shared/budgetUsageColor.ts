export function budgetUsageColor(percentage: number): string {
  if (percentage < 60) return "#16A34A";
  if (percentage < 80) return "#EAB308";
  if (percentage < 90) return "#F97316";
  if (percentage < 95) return "#F05A2E";
  return "#E5484D";
}
