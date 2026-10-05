"use client";

import { Button } from "@/components/ui/button";
import { runTime } from "../../model/format";
import { useAnalysisKeyInfo } from "./useAnalysisKeyInfo";

function budgetLabel(amount: number | null, duration?: string | null): string {
  if (amount === null) return "No key budget";
  const periods: Record<string, string> = {
    "1mo": "month",
    "30d": "month",
    "1d": "day",
    "24h": "day",
    "7d": "week",
    "1h": "hour",
  };
  const dollars = new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    maximumFractionDigits: 2,
  }).format(amount);
  return duration ? `${dollars} / ${periods[duration] ?? duration}` : `${dollars} total`;
}

function modelsLabel(models: readonly string[]): string {
  return models.length ? models.join(", ") : "All models";
}

export function AnalysisKeySummary({ keyId }: { keyId: string }) {
  const key = useAnalysisKeyInfo(keyId);
  if (key.isLoading) return <p className="text-xs text-muted-foreground">Loading billing key…</p>;
  if (key.error || !key.data)
    return (
      <p role="alert" className="flex items-center gap-2 text-xs text-destructive">
        Could not load billing key
        <Button variant="link" size="xs" className="h-auto p-0" onClick={() => void key.refetch()}>
          Retry
        </Button>
      </p>
    );
  const info = key.data;
  const summary = [
    `Bills to ${info.key_alias || "an assigned virtual key"}`,
    modelsLabel(info.models),
    budgetLabel(info.max_budget, info.budget_duration),
  ].join(" · ");
  const inactive = info.status && info.status !== "active";
  return (
    <p className="truncate text-xs text-muted-foreground">
      {summary}
      {inactive && <span className="text-destructive"> · Key {info.status}</span>}
    </p>
  );
}

export function AnalysisKeyDetails({ keyId }: { keyId: string }) {
  const key = useAnalysisKeyInfo(keyId);
  if (key.isLoading) return <p className="text-xs text-muted-foreground">Loading key permissions…</p>;
  if (key.error || !key.data)
    return (
      <div role="alert" className="flex items-center gap-2 text-sm text-destructive">
        Could not load key permissions
        <Button variant="ghost" size="sm" onClick={() => void key.refetch()}>
          Retry
        </Button>
      </div>
    );
  const info = key.data;
  return (
    <div className="space-y-2 text-sm">
      <dl className="grid grid-cols-[auto_minmax(0,1fr)] gap-x-5 gap-y-2">
        <dt className="text-muted-foreground">Models</dt>
        <dd className="break-words">{modelsLabel(info.models)}</dd>
        <dt className="text-muted-foreground">Key limit</dt>
        <dd>{budgetLabel(info.max_budget, info.budget_duration)}</dd>
      </dl>
      {info.status && info.status !== "active" && (
        <p role="alert" className="text-destructive">
          This key is {info.status}. Choose an active key.
        </p>
      )}
      <details className="text-xs text-muted-foreground">
        <summary className="cursor-pointer">Other limits</summary>
        <div className="mt-2 space-y-1 leading-5">
          <p>Requests per minute: {info.rpm_limit ?? "No key limit"}</p>
          <p>Tokens per minute: {info.tpm_limit ?? "No key limit"}</p>
          <p>Expires: {info.expires ? runTime(info.expires) : "No expiry"}</p>
          <p>Team, organization, and model limits still apply.</p>
        </div>
      </details>
    </div>
  );
}
