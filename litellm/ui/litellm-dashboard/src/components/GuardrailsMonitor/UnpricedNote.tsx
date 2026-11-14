import React from "react";
import { Trans } from "react-i18next";
import { pricingIssueUrl, totalUnits, type UsageUnits } from "./usageUnits";

export function UnpricedNote({ unpriced, provider }: { unpriced: UsageUnits; provider?: string }) {
  const total = totalUnits(unpriced);
  if (total === 0) return null;
  return (
    <p className="text-xs text-warning">
      <Trans
        i18nKey="guardrailsMonitor.unpricedNote.message"
        count={total}
        values={{ value: total.toLocaleString() }}
        components={{
          githubLink: (
            <a
              href={pricingIssueUrl(unpriced, provider)}
              target="_blank"
              rel="noreferrer"
              className="underline underline-offset-2"
            />
          ),
        }}
      />
    </p>
  );
}
