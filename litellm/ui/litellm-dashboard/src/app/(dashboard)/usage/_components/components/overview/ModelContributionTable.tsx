"use client";

import type { ParseKeys } from "i18next";
import { useTranslation } from "react-i18next";
import {
  NUMERIC_CELL_CLASS,
  Table,
  TableBody,
  TableCaption,
  TableCell,
  TableFooter,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { cn } from "@/lib/cva.config";
import { formatExactCount, formatExactUsd } from "./trendFormat";
import type { ContributionRow, ObservedRate } from "./usageTrendData";

const ONE_DECIMAL = { minimumFractionDigits: 1, maximumFractionDigits: 1 } as const;

const RATE_GAP_LABEL = {
  noTokens: "usage.trend.table.spendWithoutTokens",
  noSpend: "usage.trend.table.noMeteredSpend",
  none: "usage.trend.table.notApplicable",
} as const satisfies Record<Exclude<ObservedRate["kind"], "rate">, ParseKeys>;

function Rate({ rate }: { rate: ObservedRate }) {
  const { t } = useTranslation();
  if (rate.kind === "rate") return <>{formatExactUsd(rate.usdPerMillion)}</>;
  return <span className="text-xs text-muted-foreground">{t(RATE_GAP_LABEL[rate.kind])}</span>;
}

function Share({ value }: { value: number | null }) {
  const { t } = useTranslation();
  if (value === null) return <span className="text-muted-foreground">{t("usage.trend.table.notApplicable")}</span>;
  return <>{t("usage.trend.table.percent", { value: t("format.number", { val: value, ...ONE_DECIMAL }) })}</>;
}

function Figures({ row }: { row: ContributionRow }) {
  const { t } = useTranslation();
  return (
    <>
      <TableCell className={NUMERIC_CELL_CLASS}>{formatExactCount(row.tokens)}</TableCell>
      <TableCell className={cn(NUMERIC_CELL_CLASS, "text-muted-foreground")}>
        <Share value={row.tokenShare} />
      </TableCell>
      <TableCell className={NUMERIC_CELL_CLASS}>{formatExactUsd(row.spend)}</TableCell>
      <TableCell className={cn(NUMERIC_CELL_CLASS, "text-muted-foreground")}>
        <Share value={row.spendShare} />
      </TableCell>
      <TableCell className={NUMERIC_CELL_CLASS}>
        <Rate rate={row.rate} />
      </TableCell>
    </>
  );
}

export default function ModelContributionTable({
  rows,
  total,
}: {
  rows: readonly ContributionRow[];
  total: ContributionRow;
}) {
  const { t } = useTranslation();
  return (
    <Table>
      <TableCaption className="mt-3 text-left text-xs leading-5">{t("usage.trend.table.rateNote")}</TableCaption>
      <TableHeader>
        <TableRow>
          <TableHead>{t("usage.trend.table.model")}</TableHead>
          <TableHead className={NUMERIC_CELL_CLASS}>{t("usage.trend.table.tokens")}</TableHead>
          <TableHead className={NUMERIC_CELL_CLASS}>{t("usage.trend.table.tokenShare")}</TableHead>
          <TableHead className={NUMERIC_CELL_CLASS}>{t("usage.trend.table.spend")}</TableHead>
          <TableHead className={NUMERIC_CELL_CLASS}>{t("usage.trend.table.spendShare")}</TableHead>
          <TableHead className={NUMERIC_CELL_CLASS}>{t("usage.trend.table.observedRate")}</TableHead>
        </TableRow>
      </TableHeader>
      <TableBody>
        {rows.map((row) => (
          <TableRow key={row.key}>
            <TableCell className="max-w-80 whitespace-normal">
              <span className="flex min-w-0 items-start gap-2">
                <span
                  aria-hidden="true"
                  className="mt-1 size-2.5 shrink-0 rounded-[2px]"
                  style={{ backgroundColor: row.color }}
                />
                <span className="min-w-0 break-words font-medium text-foreground">{row.key}</span>
                {row.grouped && (
                  <span className="shrink-0 rounded bg-muted px-1.5 py-0.5 text-[11px] leading-4 text-muted-foreground">
                    {t("usage.trend.table.inOther")}
                  </span>
                )}
              </span>
            </TableCell>
            <Figures row={row} />
          </TableRow>
        ))}
      </TableBody>
      <TableFooter>
        <TableRow>
          <TableCell className="font-medium">{total.key}</TableCell>
          <Figures row={total} />
        </TableRow>
      </TableFooter>
    </Table>
  );
}
