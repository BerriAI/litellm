import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { METRIC_LABELS, formatAlertValue } from "@/components/UsageAlerts/usageAlertApi";
import type { Issue } from "./issueApi";

export default function IssueComparisonTable({ issue }: { issue: Issue }) {
  const comparisons = issue.comparisons ?? [];
  if (!comparisons.length) return null;
  return (
    <Table aria-label="Saved issue comparisons">
      <TableHeader>
        <TableRow>
          <TableHead>Metric</TableHead>
          <TableHead>Observed · {issue.observed_on} UTC</TableHead>
          <TableHead>Daily baseline</TableHead>
          <TableHead>Issue condition</TableHead>
        </TableRow>
      </TableHeader>
      <TableBody>
        {comparisons.map((comparison, index) => (
          <TableRow key={`${comparison.metric}-${index}`}>
            <TableCell className="font-medium">{METRIC_LABELS[comparison.metric]}</TableCell>
            <TableCell className="tabular-nums">
              {formatAlertValue(comparison.metric, comparison.observed_value)}
            </TableCell>
            <TableCell className="tabular-nums">
              {formatAlertValue(comparison.metric, comparison.baseline_value)}
              <p className="mt-1 text-xs text-muted-foreground">
                {comparison.baseline_start}–{comparison.baseline_end} UTC
                {comparison.baseline_sample_days ? ` · ${comparison.baseline_sample_days} recorded days` : ""}
              </p>
            </TableCell>
            <TableCell className="tabular-nums">
              {typeof comparison.threshold === "number" ? (
                <>
                  {comparison.direction === "increase" ? "At least" : "At most"}{" "}
                  {formatAlertValue(comparison.metric, comparison.threshold)}
                </>
              ) : (
                <span className="text-muted-foreground">Context only</span>
              )}
            </TableCell>
          </TableRow>
        ))}
      </TableBody>
    </Table>
  );
}
