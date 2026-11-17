"use client";

import type { ParseKeys } from "i18next";
import { useTranslation } from "react-i18next";
import { ChevronRight, Search } from "lucide-react";
import {
  type UsageMetric,
  formatCompact,
  formatUsd,
} from "@/app/(dashboard)/usage/_components/components/overview/overviewData";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { cn } from "@/lib/cva.config";
import type { CustomerUsageRow } from "./customerUsageData";
import { ModelMark } from "@/app/(dashboard)/usage/_components/components/overview/BreakdownChart";
import { colorForModel } from "@/app/(dashboard)/usage/_components/components/overview/overviewData";
import { providerForModel } from "@/app/(dashboard)/usage/_components/components/overview/modelProvider";

export const CUSTOMER_USAGE_DIMENSIONS = ["models", "api_keys", "teams"] as const;
export type CustomerUsageDimension = (typeof CUSTOMER_USAGE_DIMENSIONS)[number];

const DIMENSION_LABELS: Record<CustomerUsageDimension, ParseKeys> = {
  models: "customerPortal.analytics.models",
  api_keys: "customerPortal.analytics.keys",
  teams: "customerPortal.analytics.teams",
};
const PAGE_SIZE = 10;

interface CustomerUsageRankingsProps {
  rows: readonly CustomerUsageRow[];
  dimension: CustomerUsageDimension;
  metric: UsageMetric;
  search: string;
  page: number;
  colorByModel: ReadonlyMap<string, string>;
  onSearchChange: (search: string) => void;
  onPageChange: (page: number) => void;
  onDrillDown: (dimension: "teams", id: string) => void;
}

export default function CustomerUsageRankings({
  rows,
  dimension,
  metric,
  search,
  page,
  colorByModel,
  onSearchChange,
  onPageChange,
  onDrillDown,
}: CustomerUsageRankingsProps) {
  const { t } = useTranslation();
  const filtered = rows.filter((row) => row.label.toLocaleLowerCase().includes(search.toLocaleLowerCase()));
  const lastPage = Math.max(1, Math.ceil(filtered.length / PAGE_SIZE));
  const currentPage = Math.min(Math.max(1, page), lastPage);
  const start = (currentPage - 1) * PAGE_SIZE;
  const canDrillDown = dimension === "teams";

  return (
    <section className="min-w-0" aria-label={t("customerPortal.analytics.breakdown")}>
      <div className="flex flex-wrap items-center justify-end gap-x-5 gap-y-3">
        <div className="relative mb-2 w-full sm:w-52">
          <Search aria-hidden="true" className="absolute top-2.5 left-3 size-3.5 text-muted-foreground" />
          <Input
            aria-label={t("customerPortal.analytics.search")}
            placeholder={t("customerPortal.analytics.search")}
            className="h-8 pl-9 text-xs"
            value={search}
            onChange={(event) => onSearchChange(event.target.value)}
          />
        </div>
      </div>
      <div>
        <p className="py-3 text-xs text-muted-foreground">{t("customerPortal.analytics.rankingHint")}</p>
        <Table aria-label={t(DIMENSION_LABELS[dimension])}>
          <TableHeader>
            <TableRow>
              <TableHead className="w-10 pl-0">{t("customerPortal.analytics.rank")}</TableHead>
              <TableHead>{t(DIMENSION_LABELS[dimension])}</TableHead>
              <TableHead className="min-w-28 text-right">{t("customerPortal.analytics.share")}</TableHead>
              <TableHead className="text-right" aria-sort={metric === "spend" ? "descending" : undefined}>
                {t("customerPortal.analytics.spend")}
              </TableHead>
              <TableHead className="text-right" aria-sort={metric === "tokens" ? "descending" : undefined}>
                {t("customerPortal.analytics.tokens")}
              </TableHead>
              <TableHead className="pr-0 text-right" aria-sort={metric === "requests" ? "descending" : undefined}>
                {t("customerPortal.analytics.requests")}
              </TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {filtered.slice(start, start + PAGE_SIZE).map((row, index) => (
              <TableRow key={row.id}>
                <TableCell className="pl-0 text-xs tabular-nums text-muted-foreground">{start + index + 1}</TableCell>
                <TableCell className="max-w-72 py-4">
                  {canDrillDown && row.id ? (
                    <button
                      type="button"
                      onClick={() => onDrillDown(dimension, row.id)}
                      className="flex max-w-full items-center gap-2 text-left font-medium hover:text-primary focus-visible:outline-2 focus-visible:outline-ring"
                    >
                      <span className="truncate">{row.label}</span>
                      <ChevronRight aria-hidden="true" className="size-3.5 shrink-0 text-muted-foreground" />
                    </button>
                  ) : (
                    <span className="flex min-w-0 items-center gap-3" title={row.label}>
                      {dimension === "models" && (
                        <ModelMark model={row.id} color={colorForModel(row.id, colorByModel)} />
                      )}
                      <span className="min-w-0">
                        <span className="block truncate font-medium">{row.label}</span>
                        {dimension === "models" && providerForModel(row.id) && (
                          <span className="block text-xs text-muted-foreground">
                            {t("usage.breakdown.byProvider", { provider: providerForModel(row.id) })}
                          </span>
                        )}
                      </span>
                    </span>
                  )}
                </TableCell>
                <TableCell className="text-right">
                  <div className="flex items-center justify-end gap-3">
                    <div
                      aria-hidden="true"
                      className="hidden h-1.5 w-20 overflow-hidden rounded-full bg-muted md:block"
                    >
                      <div
                        className="h-full rounded-full bg-primary"
                        style={{
                          width: `${Math.min(100, Math.max(0, row.share * 100))}%`,
                          ...(dimension === "models" ? { backgroundColor: colorForModel(row.id, colorByModel) } : {}),
                        }}
                      />
                    </div>
                    <span className="w-12 text-xs tabular-nums">
                      {t("format.number", { val: row.share * 100, maximumFractionDigits: 1 })}%
                    </span>
                  </div>
                </TableCell>
                <TableCell className={cn("text-right tabular-nums", metric !== "spend" && "text-muted-foreground")}>
                  {formatUsd(row.spend)}
                </TableCell>
                <TableCell
                  title={row.tokens.toLocaleString()}
                  className={cn("text-right tabular-nums", metric !== "tokens" && "text-muted-foreground")}
                >
                  {formatCompact(row.tokens)}
                </TableCell>
                <TableCell
                  title={row.requests.toLocaleString()}
                  className={cn("pr-0 text-right tabular-nums", metric !== "requests" && "text-muted-foreground")}
                >
                  {formatCompact(row.requests)}
                </TableCell>
              </TableRow>
            ))}
            {filtered.length === 0 && (
              <TableRow>
                <TableCell colSpan={6} className="py-12 text-center text-muted-foreground">
                  {t(search ? "customerPortal.analytics.noMatches" : "customerPortal.analytics.noBreakdown")}
                </TableCell>
              </TableRow>
            )}
          </TableBody>
        </Table>
        {filtered.length > PAGE_SIZE && (
          <div className="flex items-center justify-between border-t pt-3 text-xs text-muted-foreground">
            <span>
              {t("customerPortal.analytics.pagination", {
                from: start + 1,
                to: Math.min(start + PAGE_SIZE, filtered.length),
                total: filtered.length,
              })}
            </span>
            <div className="flex gap-2">
              <Button
                size="sm"
                variant="outline"
                disabled={currentPage === 1}
                onClick={() => onPageChange(currentPage - 1)}
              >
                {t("customerPortal.analytics.previous")}
              </Button>
              <Button
                size="sm"
                variant="outline"
                disabled={currentPage === lastPage}
                onClick={() => onPageChange(currentPage + 1)}
              >
                {t("customerPortal.analytics.next")}
              </Button>
            </div>
          </div>
        )}
      </div>
    </section>
  );
}
