import { useTranslation } from "react-i18next";

const cacheCurrencyFormat = { style: "currency", currency: "USD", maximumFractionDigits: 6 } as const;

export default function CustomerCacheCost({
  usage,
}: {
  usage: {
    cacheReadTokens: number;
    cacheWriteTokens: number;
    cacheCost: number;
    pricedCacheTokens: number;
  } | null;
}) {
  const { t } = useTranslation();
  if (!usage) return "—";
  const hasCacheTokens = usage.cacheReadTokens > 0 || usage.cacheWriteTokens > 0;
  if (usage.cacheCost === 0 && usage.pricedCacheTokens === 0 && hasCacheTokens) {
    return "—";
  }
  const format = (val: number) => t("format.number", { val, ...cacheCurrencyFormat });
  return (
    <span>{usage.cacheCost > 0 && usage.cacheCost < 0.000001 ? `<${format(0.000001)}` : format(usage.cacheCost)}</span>
  );
}
