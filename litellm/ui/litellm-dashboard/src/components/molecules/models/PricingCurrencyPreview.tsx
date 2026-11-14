import type { ParseKeys } from "i18next";
import { useTranslation } from "react-i18next";

import { formatAmount, type PricePreviewRow, type PriceUnit, type PricingCurrency } from "@/utils/pricingCurrency";

const UNIT_LABELS: Record<PriceUnit, ParseKeys> = {
  million_tokens: "pricingCurrency.units.million_tokens",
  million_characters: "pricingCurrency.units.million_characters",
  second: "pricingCurrency.units.second",
  image: "pricingCurrency.units.image",
  query: "pricingCurrency.units.query",
  page: "pricingCurrency.units.page",
  pixel: "pricingCurrency.units.pixel",
  ptu_hour: "pricingCurrency.units.ptu_hour",
  clip: "pricingCurrency.units.clip",
  unit: "pricingCurrency.units.unit",
};

interface PricingCurrencyPreviewProps {
  currency: PricingCurrency;
  rows: PricePreviewRow[];
  reinterpreted: string[];
  rateValid: boolean;
}

export function PricingCurrencyPreview({ currency, rows, reinterpreted, rateValid }: PricingCurrencyPreviewProps) {
  const { t } = useTranslation();
  if (rows.length === 0) {
    return (
      <p className="text-xs text-muted-foreground">
        {currency === "USD"
          ? t("pricingCurrency.preview.empty")
          : t("pricingCurrency.preview.emptyCurrency", { currency })}
      </p>
    );
  }
  return (
    <div className="space-y-2" aria-live="polite">
      {reinterpreted.length > 0 && (
        <p role="status" className="rounded-sm border border-amber-300 bg-amber-50 p-2 text-xs text-amber-900">
          {t("pricingCurrency.preview.reinterpreted", { currency, fields: reinterpreted.join(", ") })}
        </p>
      )}
      <table className="w-full text-left text-xs">
        <caption className="mb-1 text-left text-xs font-medium text-foreground">
          {t("pricingCurrency.preview.caption", { currency })}
        </caption>
        <thead className="text-muted-foreground">
          <tr>
            <th scope="col" className="py-1 pr-2 font-medium">
              {t("pricingCurrency.preview.field")}
            </th>
            <th scope="col" className="py-1 pr-2 font-medium">
              {t("pricingCurrency.preview.configured", { currency })}
            </th>
            <th scope="col" className="py-1 font-medium">
              {t("pricingCurrency.preview.usd")}
            </th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => {
            const unit = t(UNIT_LABELS[row.unit]);
            return (
              <tr key={row.field} className="border-t border-border">
                <td className="py-1 pr-2 font-mono">{row.field}</td>
                <td className="py-1 pr-2">
                  {t("pricingCurrency.preview.amount", { currency, amount: formatAmount(row.source), unit })}
                </td>
                <td className="py-1">
                  {row.usd === null || !rateValid
                    ? t("pricingCurrency.preview.needsRate")
                    : t("pricingCurrency.preview.amount", { currency: "USD", amount: formatAmount(row.usd), unit })}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
