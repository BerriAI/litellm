import { Badge } from "@/components/ui/badge";
import { CircleAlert, CircleCheck, TriangleAlert } from "lucide-react";
import { errorCount as countErrors, type PayloadValidation } from "./lib/validatePayload";

export function ValidationStatus({ validation }: { validation: PayloadValidation<unknown> }) {
  const errorCount = countErrors(validation);
  if (errorCount > 0) {
    return (
      <Badge variant="destructive">
        {errorCount} {errorCount === 1 ? "issue" : "issues"}
      </Badge>
    );
  }
  return <Badge variant="secondary">Valid payload</Badge>;
}

export function IssueList({ id, validation }: { id: string; validation: PayloadValidation<unknown> }) {
  if (validation.issues.length === 0) {
    return (
      <p id={id} role="status" className="flex items-center gap-1.5 border-t px-3 py-2 text-xs text-muted-foreground">
        <CircleCheck className="size-3.5 text-emerald-600 dark:text-emerald-400" />
        Ready to send
      </p>
    );
  }
  return (
    <ul id={id} aria-label="Payload validation issues" className="grid max-h-36 gap-1 overflow-auto border-t px-3 py-2">
      {validation.issues.map((issue, index) => (
        <li key={`${issue.path}-${index}`} className="flex items-start gap-1.5 text-xs">
          {issue.severity === "error" ? (
            <CircleAlert className="mt-0.5 size-3.5 shrink-0 text-destructive" />
          ) : (
            <TriangleAlert className="mt-0.5 size-3.5 shrink-0 text-amber-600 dark:text-amber-400" />
          )}
          <code className="shrink-0 rounded bg-muted px-1 font-mono">{issue.path}</code>
          <span>{issue.message}</span>
        </li>
      ))}
    </ul>
  );
}
