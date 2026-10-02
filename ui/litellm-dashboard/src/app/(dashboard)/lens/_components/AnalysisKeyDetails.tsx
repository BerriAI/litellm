"use client";

import { useQuery } from "@tanstack/react-query";
import { z } from "zod";
import { apiClient } from "@/components/networking";
import { Button } from "@/components/ui/button";
import { runTime } from "./lensData";

const keyInfoFields = {
  key_alias: z.string().nullable().optional(),
  models: z.array(z.string()),
  max_budget: z.number().nullable(),
  budget_duration: z.string().nullable().optional(),
  rpm_limit: z.number().nullable().optional(),
  tpm_limit: z.number().nullable().optional(),
  expires: z.string().nullable().optional(),
  status: z.string().optional(),
};
const keyInfoSchema = z.object({ info: z.object(keyInfoFields) });

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

export function useAnalysisKeyInfo(accessToken: string, keyId?: string) {
  return useQuery({
    queryKey: ["lens-key-info", accessToken, keyId],
    enabled: !!keyId,
    queryFn: async () =>
      keyInfoSchema.parse(await apiClient.get("/key/info", { accessToken, query: { key: keyId } })).info,
  });
}

export function AnalysisKeyDetails({
  accessToken,
  keyId,
  showName = false,
}: {
  accessToken: string;
  keyId: string;
  showName?: boolean;
}) {
  const key = useAnalysisKeyInfo(accessToken, keyId);
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
        {showName && (
          <>
            <dt className="text-muted-foreground">Billing key</dt>
            <dd className="break-words">{info.key_alias || "Assigned virtual key"}</dd>
          </>
        )}
        <dt className="text-muted-foreground">Models</dt>
        <dd className="break-words">{info.models.length ? info.models.join(", ") : "All models"}</dd>
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
