import { useLanguage } from "@/contexts/LanguageContext";
import React from "react";
import { pricingIssueUrl, totalUnits, type UsageUnits } from "./usageUnits";

export function UnpricedNote({ unpriced, provider }: { unpriced: UsageUnits; provider?: string }) {
  const { t } = useLanguage();
  const total = totalUnits(unpriced);
  if (total === 0) return null;
  return (
    <p className="text-xs text-warning">
      {/* 完整句子保留英文单复数，并允许中文调整语序。 */}
      {t(
        total === 1
          ? "{count} unit with no known price is left out of the cost."
          : "{count} units with no known price are left out of the cost.",
        { count: total.toLocaleString() },
      )}{" "}
      <a
        href={pricingIssueUrl(unpriced, provider)}
        target="_blank"
        rel="noreferrer"
        className="underline underline-offset-2"
      >
        {t("Request pricing on GitHub")}
      </a>
    </p>
  );
}
