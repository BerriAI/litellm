"use client";

import { useNow } from "@/hooks/useNow";
import { durationLabel, money, scopeLabel } from "../../model/format";
import { nextCheckStatus } from "../../model/status";
import { type Lens } from "../../model/types";

export function InvestigationSummary({ lens }: { lens: Lens }) {
  const now = useNow(15000);
  const { settings } = lens;
  const spent = lens.budget_month === new Date(now).toISOString().slice(0, 7) ? lens.spent ?? 0 : 0;
  const parts = [
    settings.enabled ? `Every ${durationLabel(settings.interval_minutes)}` : "One-off",
    nextCheckStatus(lens, now),
    `${money(spent)} of ${money(settings.monthly_budget ?? 100)} this month`,
  ].filter(Boolean);
  return (
    <p className="mt-1 text-xs text-muted-foreground">
      <span className="font-mono">{scopeLabel(settings)}</span>
      {parts.map((part) => ` · ${part}`)}
    </p>
  );
}
