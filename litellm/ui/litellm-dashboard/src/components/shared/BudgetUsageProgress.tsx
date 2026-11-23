import { budgetUsageColor } from "./budgetUsageColor";

export function BudgetUsageProgress({
  percentage,
  label,
  valueText,
}: {
  percentage: number;
  label: string;
  valueText: string;
}) {
  const value = Math.max(0, Math.min(100, percentage));
  return (
    <div
      role="progressbar"
      aria-label={label}
      aria-valuemin={0}
      aria-valuemax={100}
      aria-valuenow={value}
      aria-valuetext={valueText}
      className="h-1 overflow-hidden rounded-full bg-muted"
    >
      <div
        className="h-full rounded-full"
        style={{
          backgroundColor: budgetUsageColor(value),
          width: `${value}%`,
        }}
      />
    </div>
  );
}
