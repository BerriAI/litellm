"use client";

import React, { useMemo } from "react";
import { ArrowDown, ArrowUp, ArrowUpDown, Info } from "lucide-react";
import { parseAsStringLiteral, useQueryStates } from "nuqs";

import AdvancedDatePicker from "@/components/shared/advanced_date_picker";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";
import { formatNumberWithCommas } from "@/utils/dataUtils";
import { CacheLeakageDimension, CacheLeakageRow, computeCacheLeakage, pct, usd } from "./costOptimizationUtils";
import { DailyActivityRange } from "./useDailyActivityRange";

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
  const active = sort.column === column;
  const ActiveArrow = sort.dir === "asc" ? ArrowUp : ArrowDown;
  const Arrow = active ? ActiveArrow : ArrowUpDown;
  return (
    <TableHead className="text-right">
      <span className="inline-flex items-center justify-end gap-1">
        <button
          type="button"
          onClick={() => onSort(column)}
          aria-label={`Sort by ${label}`}
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
  const { dateValue, onDateChange, results, loading, isFetchingMore } = activity;
  const [{ leak_by: dimension, leak_sort: sortColumn, leak_dir: sortDir }, setLeakageState] =
    useQueryStates(LEAKAGE_URL_STATE);
  const sort = useMemo<SortState>(() => ({ column: sortColumn, dir: sortDir }), [sortColumn, sortDir]);
  const leakage = useMemo(() => computeCacheLeakage(results, dimension), [results, dimension]);
  const rows = useMemo(() => [...leakage.rows].sort((a, b) => compareRows(a, b, sort)), [leakage.rows, sort]);

  const onSort = (column: SortColumn) =>
    void setLeakageState((prev) => ({
      leak_sort: column,
      leak_dir: prev.leak_sort === column ? flipped(prev.leak_dir) : NATURAL_DIR[column],
    }));

  const subject = dimension === "model" ? "Models" : "Keys";
  const firstColumn = dimension === "model" ? "Model" : "Key";
  const emptyNoun = dimension === "model" ? "model" : "key";

  return (
    <TooltipProvider delay={300}>
      <Card>
        <CardHeader>
          <div className="flex flex-col gap-4 md:flex-row md:items-start md:justify-between">
            <div className="min-w-0">
              <CardTitle>Cache leakage by {dimension === "model" ? "model" : "virtual key"}</CardTitle>
              <p className="mt-1 text-sm text-muted-foreground line-clamp-2">
                {subject} sending large volumes of uncached input with a low cache hit rate are likely missing prompt
                caching. Potential savings is approximate: uncached input priced at what your cached traffic nets per
                cached token, after cache-write premiums.
              </p>
            </div>
            <div className="shrink-0">
              <AdvancedDatePicker value={dateValue} onValueChange={onDateChange} />
            </div>
          </div>
          <Tabs
            value={dimension}
            onValueChange={(value) => void setLeakageState({ leak_by: value === "model" ? "model" : "key" })}
          >
            <TabsList>
              <TabsTrigger value="key">By virtual key</TabsTrigger>
              <TabsTrigger value="model">By model</TabsTrigger>
            </TabsList>
          </Tabs>
        </CardHeader>
        <CardContent>
          {rows.length > 0 && isFetchingMore && (
            <p className="mb-2 text-sm text-muted-foreground">
              Data is still loading; rows and totals will update as the rest of the range arrives.
            </p>
          )}
          {rows.length === 0 ? (
            <p className="py-8 text-center text-sm text-muted-foreground">
              {loading || isFetchingMore ? "Loading..." : `No ${emptyNoun} usage in this range.`}
            </p>
          ) : (
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>{firstColumn}</TableHead>
                  <SortableHead
                    column="uncachedPromptTokens"
                    label="Uncached input tokens"
                    info="Input tokens you sent in this range that weren't served from or written to the cache"
                    sort={sort}
                    onSort={onSort}
                  />
                  <SortableHead
                    column="cacheHitRatio"
                    label="Cache hit rate"
                    info="Share of your input tokens that were served from the cache"
                    sort={sort}
                    onSort={onSort}
                  />
                  <SortableHead
                    column="potentialSavings"
                    label="Potential savings"
                    info="About how much you'd save if this uncached input used prompt caching. Estimated as uncached input tokens times what your cached traffic already nets per cached token (realized cache savings, after write premiums, ÷ cache read and write tokens). Blank when caching is not currently saving anything overall."
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
