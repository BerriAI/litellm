import { useId, useMemo } from "react";
import { useTranslation } from "react-i18next";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import type { DailyData } from "@/components/UsagePage/types";
import { customerModelTokenUsage } from "./customerUsageTotals";
import CustomerCacheCost from "./CustomerCacheCost";

const SEGMENTS = [
  { id: "cacheReadTokens", label: "customerPortal.tokenUsage.cacheReadTokens", color: "bg-blue-600 dark:bg-blue-400" },
  {
    id: "cacheWriteTokens",
    label: "customerPortal.tokenUsage.cacheWriteTokens",
    color: "bg-violet-500 dark:bg-violet-400",
  },
  { id: "other", label: "customerPortal.tokenUsage.otherTokens", color: "bg-slate-300 dark:bg-slate-600" },
] as const;
const MODEL_LIMIT = 10;

export default function CustomerModelTokenChart({ days }: { days: readonly DailyData[] }) {
  const { t } = useTranslation();
  const headingId = useId();
  const models = useMemo(() => customerModelTokenUsage(days), [days]);
  const max = models[0]?.tokens;
  if (!max) return null;
  const displayed = models.slice(0, MODEL_LIMIT);

  return (
    <section aria-labelledby={headingId} className="space-y-2 rounded-lg border bg-card p-4 sm:p-5">
      <h2 id={headingId} className="text-base font-medium tracking-tight">
        {t("customerPortal.tokenUsage.modelTitle")}
      </h2>
      {models.length > MODEL_LIMIT && (
        <p className="text-xs text-muted-foreground">
          {t("customerPortal.tokenUsage.topModels", { limit: MODEL_LIMIT })}
        </p>
      )}
      <Table aria-labelledby={headingId}>
        <TableHeader>
          <TableRow>
            <TableHead scope="col" className="pl-0">
              {t("customerPortal.analytics.models")}
            </TableHead>
            <TableHead scope="col" className="w-full min-w-32">
              {t("customerPortal.tokenUsage.volume")}
            </TableHead>
            {SEGMENTS.map(({ id, label, color }) => (
              <TableHead scope="col" key={id} className="text-right">
                <span aria-hidden="true" className={`mr-1.5 inline-block size-2.5 rounded-sm ${color}`} />
                {t(label)}
              </TableHead>
            ))}
            <TableHead scope="col" className="pr-0 text-right">
              {t("usage.statStrip.tokenPopover.totalTokens")}
            </TableHead>
            <TableHead scope="col" className="text-right">
              {t("customerPortal.tokenUsage.cacheCost")}
            </TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {displayed.map((row) => (
            <TableRow key={row.model}>
              <th scope="row" className="py-3 pr-2 text-left font-medium">
                <span className="block max-w-56 truncate" title={row.model}>
                  {row.model}
                </span>
              </th>
              <TableCell>
                <div
                  aria-hidden="true"
                  className="flex h-3 overflow-hidden rounded-sm"
                  style={{ width: `${(row.tokens / max) * 100}%` }}
                >
                  {row.other === null ? (
                    <span className="w-full bg-muted-foreground/30" />
                  ) : (
                    SEGMENTS.map(({ id, color }) => (
                      <span key={id} className={color} style={{ width: `${((row[id] ?? 0) / row.tokens) * 100}%` }} />
                    ))
                  )}
                </div>
              </TableCell>
              {SEGMENTS.map(({ id }) => (
                <TableCell key={id} className="text-right tabular-nums">
                  {row[id] === null ? "—" : t("format.number", { val: row[id] })}
                </TableCell>
              ))}
              <TableCell className="pr-0 text-right font-medium tabular-nums">
                {t("format.number", { val: row.tokens })}
              </TableCell>
              <TableCell className="text-right tabular-nums">
                <CustomerCacheCost usage={row} />
              </TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
      {displayed.some((row) => row.other === null) && (
        <p className="text-xs text-muted-foreground">{t("customerPortal.tokenUsage.inconsistent")}</p>
      )}
    </section>
  );
}
