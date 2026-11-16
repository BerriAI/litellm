"use client";

import { useMemo, useState } from "react";
import { useTranslation } from "react-i18next";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { useReferenceRates } from "@/app/(dashboard)/hooks/referenceRates/useReferenceRates";
import { estimatedSpend } from "./spendDisplay";

interface SpendDisplayCurrencyProps {
  usdSpend: number;
  loading: boolean;
}

/** Shows the USD spend total estimated in another currency at the current ECB reference rate. */
export default function SpendDisplayCurrency({ usdSpend, loading }: SpendDisplayCurrencyProps) {
  const { t, i18n } = useTranslation();
  const [requested, setRequested] = useState(false);
  const [currency, setCurrency] = useState("USD");
  const rates = useReferenceRates(requested);
  const usable = rates.isError ? undefined : rates.data;
  const names = useMemo(
    () => new Intl.DisplayNames([i18n.language], { type: "currency", fallback: "code" }),
    [i18n.language],
  );
  const codes = Object.keys(usable?.usd_per_unit ?? { USD: 1 }).sort();
  const estimate =
    usable === undefined || currency === "USD"
      ? null
      : estimatedSpend(usdSpend, usable.usd_per_unit, usable.rate_date, currency, new Date());
  const shown = estimate === null ? "USD" : currency;
  const unavailable = rates.isError || (currency !== "USD" && estimate === null && !rates.isFetching);
  return (
    <div className="mt-1 grid gap-1 text-xs leading-4 text-muted-foreground">
      <div className="flex items-center gap-2">
        <span>{t("usage.statStrip.spendDisplay.label")}</span>
        <Select
          value={shown}
          onValueChange={(value) => setCurrency(value ?? "USD")}
          onOpenChange={(open) => {
            if (open) setRequested(true);
          }}
        >
          <SelectTrigger size="sm" aria-label={t("usage.statStrip.spendDisplay.label")}>
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {codes.map((code) => (
              <SelectItem key={code} value={code}>
                {t("usage.statStrip.spendDisplay.option", { code, name: names.of(code) ?? code })}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>
      {estimate !== null && !loading && (
        <p className="tabular-nums">
          {t("usage.statStrip.spendDisplay.estimate", {
            amount: new Intl.NumberFormat(i18n.language, {
              style: "currency",
              currency,
              currencyDisplay: "code",
            }).format(estimate.amount),
            date: estimate.rateDate,
          })}
        </p>
      )}
      {unavailable && <p>{t("usage.statStrip.spendDisplay.unavailable")}</p>}
    </div>
  );
}
