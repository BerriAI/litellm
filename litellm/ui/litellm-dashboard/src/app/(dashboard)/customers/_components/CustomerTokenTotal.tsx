import { useTranslation } from "react-i18next";
import { Popover, PopoverContent, PopoverTitle, PopoverTrigger } from "@/components/ui/popover";
import { cn } from "@/lib/cva.config";
import CustomerCacheCost from "./CustomerCacheCost";
import type { CustomerUsageTotal } from "./customerUsageTotals";

const TOKEN_BREAKDOWN = [
  { id: "inputTokens", label: "usage.statStrip.tokenPopover.inputTokens", nested: false },
  { id: "cacheReadTokens", label: "customerPortal.tokenUsage.cacheReadTokens", nested: true },
  { id: "cacheWriteTokens", label: "customerPortal.tokenUsage.cacheWriteTokens", nested: true },
  { id: "outputTokens", label: "usage.statStrip.tokenPopover.outputTokens", nested: false },
] as const;

export default function CustomerTokenTotal({ usage }: { usage: CustomerUsageTotal | null }) {
  const { t } = useTranslation();
  if (!usage) return "—";
  const total = t("format.number", { val: usage.tokens });
  const cacheReadRatio = usage.inputTokens > 0 ? usage.cacheReadTokens / usage.inputTokens : null;
  const percentCached =
    cacheReadRatio !== null && cacheReadRatio >= 0 && cacheReadRatio <= 1
      ? t("customerPortal.tokenUsage.percentCached", {
          percent: t("format.number", { val: cacheReadRatio, style: "percent", maximumFractionDigits: 0 }),
        })
      : undefined;
  return (
    <Popover>
      <PopoverTrigger
        openOnHover
        delay={200}
        aria-label={t("customerPortal.tokenUsage.viewTokenBreakdown", { total })}
        aria-description={percentCached}
        className="rounded-sm text-right focus-visible:ring-2 focus-visible:ring-ring"
      >
        <span className="underline decoration-muted-foreground/60 decoration-dotted underline-offset-4">
          {t("format.number", { val: usage.tokens, notation: "compact", maximumFractionDigits: 2 })}
        </span>
        {percentCached && <span className="block text-xs text-muted-foreground">{percentCached}</span>}
      </PopoverTrigger>
      <PopoverContent align="end" className="gap-2">
        <PopoverTitle>{t("customerPortal.tokenUsage.tokenBreakdown")}</PopoverTitle>
        <dl className="space-y-1 tabular-nums">
          {TOKEN_BREAKDOWN.map(({ id, label, nested }) => (
            <div key={id} className="flex justify-between gap-3">
              <dt className={cn("text-muted-foreground", nested && "pl-3")}>{t(label)}</dt>
              <dd>{t("format.number", { val: usage[id] })}</dd>
            </div>
          ))}
          <div className="mt-2 flex justify-between gap-3 border-t border-border pt-2">
            <dt className="text-muted-foreground">{t("customerPortal.tokenUsage.cacheCost")}</dt>
            <dd className="text-right">
              <CustomerCacheCost usage={usage} />
            </dd>
          </div>
        </dl>
      </PopoverContent>
    </Popover>
  );
}
