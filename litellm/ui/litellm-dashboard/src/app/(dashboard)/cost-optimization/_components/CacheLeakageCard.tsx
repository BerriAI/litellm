"use client";

import React, { useMemo } from "react";
import { useTranslation } from "react-i18next";
import { ArrowDown, ArrowUp, ArrowUpDown, Info } from "lucide-react";
import { parseAsStringLiteral, useQueryStates } from "nuqs";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";
import { formatNumberWithCommas } from "@/utils/dataUtils";
import {
  CacheLeakageDimension,
  CacheLeakageRow,
  computeCacheLeakage,
  leakageRowsFromKeyRows,
  netSavingsPerCachedToken,
  pct,
  usd,
} from "./costOptimizationUtils";
import { DailyActivityRange } from "./useDailyActivityRange";
import { useCacheLeakageKeys } from "./useCacheLeakageKeys";

interface CacheLeakageCardProps {
  activity: DailyActivityRange;
}

const SORT_COLUMNS = ["uncachedPromptTokens", "cacheHitRatio", "potentialSavings"] as const;
type SortColumn = (typeof SORT_COLUMNS)[number];
const SORT_DIRS = ["asc", "desc"] as const;
type SortDir = (typeof SORT_DIRS)[number];
interface SortState {
  column: SortColumn;
  dir: SortDir;
}

const DIMENSIONS: readonly CacheLeakageDimension[] = ["key", "model"];

const LEAKAGE_URL_STATE = {
  leak_by: parseAsStringLiteral(DIMENSIONS).withDefault("key"),
  leak_sort: parseAsStringLiteral(SORT_COLUMNS).withDefault("potentialSavings"),
  leak_dir: parseAsStringLiteral(SORT_DIRS).withDefault("desc"),
};

const NATURAL_DIR: Record<SortColumn, SortDir> = {
  uncachedPromptTokens: "desc",
  cacheHitRatio: "asc",
  potentialSavings: "desc",
};

const flipped = (dir: SortDir): SortDir => (dir === "asc" ? "desc" : "asc");

const compareRows = (a: CacheLeakageRow, b: CacheLeakageRow, sort: SortState): number => {
  const av = a[sort.column];
  const bv = b[sort.column];
  if (av == null && bv == null) return 0;
  if (av == null) return 1;
  if (bv == null) return -1;
  return sort.dir === "asc" ? av - bv : bv - av;
};

const InfoTooltip = ({ info }: { info: string }) => (
  <Tooltip>
    <TooltipTrigger render={<span className="inline-flex" aria-label={info} />}>
      <Info className="h-3 w-3 text-muted-foreground" />
    </TooltipTrigger>
    <TooltipContent className="max-w-xs">{info}</TooltipContent>
  </Tooltip>
);

const SortableHead = ({
  column,
  label,
  info,
  sort,
  onSort,
}: {
  column: SortColumn;
  label: string;
  info: string;
  sort: SortState;
  onSort: (column: SortColumn) => void;
}) => {
  const { t } = useTranslation();
  const active = sort.column === column;
  const ActiveArrow = sort.dir === "asc" ? ArrowUp : ArrowDown;
  const Arrow = active ? ActiveArrow : ArrowUpDown;
  return (
    <TableHead className="text-right">
      <span className="inline-flex items-center justify-end gap-1">
        <button
          type="button"
          onClick={() => onSort(column)}
          aria-label={t("costOptimization.cacheLeakage.sortBy", { label })}
          className="inline-flex items-center gap-1 font-medium hover:text-foreground"
        >
          {label}
          <Arrow className={`h-3 w-3 ${active ? "text-foreground" : "text-muted-foreground"}`} />
        </button>
        <InfoTooltip info={info} />
      </span>
    </TableHead>
  );
};

const CacheLeakageCard: React.FC<CacheLeakageCardProps> = ({ activity }) => {
  const { t } = useTranslation();
  const { results, loading } = activity;
  const [{ leak_by: dimension, leak_sort: sortColumn, leak_dir: sortDir }, setLeakageState] =
    useQueryStates(LEAKAGE_URL_STATE);
  const sort = useMemo<SortState>(() => ({ column: sortColumn, dir: sortDir }), [sortColumn, sortDir]);
  const leakageRate = useMemo(() => netSavingsPerCachedToken(results), [results]);
  const keyLeakage = useCacheLeakageKeys(activity, dimension === "key");
  const unsortedRows = useMemo(
    () =>
      dimension === "key"
        ? leakageRowsFromKeyRows(keyLeakage.rows, leakageRate)
        : computeCacheLeakage(results, "model").rows,
    [dimension, keyLeakage.rows, leakageRate, results],
  );
  const rows = useMemo(() => [...unsortedRows].sort((a, b) => compareRows(a, b, sort)), [unsortedRows, sort]);
  const rowsLoading = dimension === "key" ? keyLeakage.loading : loading;

  const onSort = (column: SortColumn) =>
    void setLeakageState((prev) => ({
      leak_sort: column,
      leak_dir: prev.leak_sort === column ? flipped(prev.leak_dir) : NATURAL_DIR[column],
    }));

  const subject =
    dimension === "model"
      ? t("costOptimization.cacheLeakage.subjects.models")
      : t("costOptimization.cacheLeakage.subjects.keys");
  const firstColumn =
    dimension === "model"
      ? t("costOptimization.cacheLeakage.columns.model")
      : t("costOptimization.cacheLeakage.columns.key");
  const emptyNoun =
    dimension === "model"
      ? t("costOptimization.cacheLeakage.nouns.model")
      : t("costOptimization.cacheLeakage.nouns.key");
  const emptyMessage =
    dimension === "key" && keyLeakage.failed
      ? t("costOptimization.cacheLeakage.keyLoadFailed")
      : t("costOptimization.cacheLeakage.empty", { noun: emptyNoun });

  return (
    <TooltipProvider delay={300}>
      <Card>
        <CardHeader>
          <div className="flex flex-col gap-4 md:flex-row md:items-start md:justify-between">
            <div className="min-w-0">
              <CardTitle>
                {t("costOptimization.cacheLeakage.title", {
                  dimension:
                    dimension === "model"
                      ? t("costOptimization.cacheLeakage.nouns.model")
                      : t("costOptimization.cacheLeakage.nouns.virtualKey"),
                })}
              </CardTitle>
              <p className="mt-1 text-sm text-muted-foreground line-clamp-2">
                {t("costOptimization.cacheLeakage.description", { subject })}
              </p>
            </div>
          </div>
          <Tabs
            value={dimension}
            onValueChange={(value) => void setLeakageState({ leak_by: value === "model" ? "model" : "key" })}
          >
            <TabsList>
              <TabsTrigger value="key">{t("costOptimization.cacheLeakage.tabs.byKey")}</TabsTrigger>
              <TabsTrigger value="model">{t("costOptimization.cacheLeakage.tabs.byModel")}</TabsTrigger>
            </TabsList>
          </Tabs>
        </CardHeader>
        <CardContent>
          {rows.length === 0 ? (
            <p className="py-8 text-center text-sm text-muted-foreground">
              {rowsLoading ? t("costOptimization.cacheLeakage.loading") : emptyMessage}
            </p>
          ) : (
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>{firstColumn}</TableHead>
                  <SortableHead
                    column="uncachedPromptTokens"
                    label={t("costOptimization.cacheLeakage.columns.uncachedTokens.label")}
                    info={t("costOptimization.cacheLeakage.columns.uncachedTokens.info")}
                    sort={sort}
                    onSort={onSort}
                  />
                  <SortableHead
                    column="cacheHitRatio"
                    label={t("costOptimization.cacheLeakage.columns.cacheHitRate.label")}
                    info={t("costOptimization.cacheLeakage.columns.cacheHitRate.info")}
                    sort={sort}
                    onSort={onSort}
                  />
                  <SortableHead
                    column="potentialSavings"
                    label={t("costOptimization.cacheLeakage.columns.potentialSavings.label")}
                    info={t("costOptimization.cacheLeakage.columns.potentialSavings.info")}
                    sort={sort}
                    onSort={onSort}
                  />
                </TableRow>
              </TableHeader>
              <TableBody>
                {rows.map((row) => (
                  <TableRow key={row.id}>
                    <TableCell className="font-medium">
                      {row.label}
                      {row.sublabel && <span className="ml-1 text-xs text-muted-foreground">({row.sublabel})</span>}
                    </TableCell>
                    <TableCell className="text-right">{formatNumberWithCommas(row.uncachedPromptTokens)}</TableCell>
                    <TableCell className="text-right">{pct(row.cacheHitRatio)}</TableCell>
                    <TableCell className="text-right">
                      {row.potentialSavings == null ? "—" : usd(row.potentialSavings)}
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          )}
        </CardContent>
      </Card>
    </TooltipProvider>
  );
};

export default CacheLeakageCard;
